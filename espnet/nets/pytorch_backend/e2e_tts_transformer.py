#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging

import torch
import torch.nn.functional as F

from espnet.nets.pytorch_backend.e2e_asr_transformer import subsequent_mask
from espnet.nets.pytorch_backend.e2e_tts_tacotron2 import GuidedAttentionLoss
from espnet.nets.pytorch_backend.e2e_tts_tacotron2 import make_non_pad_mask
from espnet.nets.pytorch_backend.tacotron2.decoder import Postnet
from espnet.nets.pytorch_backend.tacotron2.decoder import Prenet as DecoderPrenet
from espnet.nets.pytorch_backend.tacotron2.encoder import Encoder as EncoderPrenet
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.decoder import Decoder
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.embedding import ScaledPositionalEncoding
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.plot import _plot_and_save_attention
from espnet.nets.pytorch_backend.transformer.plot import PlotAttentionReport
from espnet.nets.tts_interface import TTSInterface
from espnet.utils.cli_utils import strtobool


class GuidedMultiHeadAttentionLoss(GuidedAttentionLoss):
    """Guided attention loss for multi head attention

    :param float sigma: standard deviation to control how close attention to a diagonal
    :param bool reset_always: whether to always reset mask
    """

    def forward(self, att_ws, ilens, olens):
        """Calculate guided attention loss for multi head attention

        :param torch.Tenosr att_ws: attention weights (B, H, T_max_out, T_max_in)
        :param torch.Tensor ilens: batch of input lenghts (B,)
        :param torch.Tensor olens: batch of output lenghts (B,)
        :return torch.tensor: guided attention loss value
        """
        if self.guided_attn_masks is None:
            self.guided_attn_masks = self._make_guided_attention_masks(ilens, olens).to(att_ws.device).unsqueeze(1)
        if self.masks is None:
            self.masks = self._make_masks(ilens, olens).to(att_ws.device).unsqueeze(1)
        losses = self.guided_attn_masks * att_ws
        loss = torch.mean(losses.masked_select(self.masks))
        if self.reset_always:
            self.reset_masks()

        return loss


class TransformerLoss(torch.nn.Module):
    """Transformer loss function

    :param Namespace args: argments containing following attributes
        (bool) use_masking: whether to mask padded part in loss calculation
        (float) bce_pos_weight: weight of positive sample of stop token (only for use_masking=True)
    """

    def __init__(self, args):
        super(TransformerLoss, self).__init__()
        self.use_masking = args.use_masking
        self.bce_pos_weight = args.bce_pos_weight

    def forward(self, after_outs, before_outs, logits, ys, labels, olens):
        """Calculate trasnformer loss

        :param torch.Tensor after_outs: outputs with postnets (B, Lmax, odim)
        :param torch.Tensor before_outs: outputs without postnets (B, Lmax, odim)
        :param torch.Tensor logits: stop logits (B, Lmax)
        :param torch.Tensor ys: batch of padded target features (B, Lmax, odim)
        :param torch.Tensor labels: batch of the sequences of stop token labels (B, Lmax)
        :param list olens: batch of the lengths of each target (B)
        :return: l1 loss value
        :rtype: torch.Tensor
        :return: mean square error loss value
        :rtype: torch.Tensor
        :return: binary cross entropy loss value
        :rtype: torch.Tensor
        """
        # perform masking for padded values
        if self.use_masking:
            mask = make_non_pad_mask(olens).unsqueeze(-1).to(ys.device)
            ys = ys.masked_select(mask)
            after_outs = after_outs.masked_select(mask)
            before_outs = before_outs.masked_select(mask)
            labels = labels.masked_select(mask[:, :, 0])
            logits = logits.masked_select(mask[:, :, 0])

        # calculate loss
        l1_loss = F.l1_loss(after_outs, ys) + F.l1_loss(before_outs, ys)
        l2_loss = F.mse_loss(after_outs, ys) + F.mse_loss(before_outs, ys)
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, labels, pos_weight=torch.tensor(self.bce_pos_weight, device=ys.device))

        return l1_loss, l2_loss, bce_loss


class TTSPlot(PlotAttentionReport):
    def plotfn(self, data, attn_dict, outdir, suffix="png", savefn=None):
        """Plot multi head attentions

        :param dict data: utts info from json file
        :param dict[str, torch.Tensor] attn_dict: multi head attention dict.
            values should be torch.Tensor (head, input_length, output_length)
        :param str outdir: dir to save fig
        :param str suffix: filename suffix including image type (e.g., png)
        :param savefn: function to save
        """
        import matplotlib.pyplot as plt
        for name, att_ws in attn_dict.items():
            for idx, att_w in enumerate(att_ws):
                filename = "%s/%s.%s.%s" % (
                    outdir, data[idx][0], name, suffix)
                if "fbank" in name:
                    fig = plt.Figure()
                    ax = fig.subplots(1, 1)
                    ax.imshow(att_w, aspect="auto")
                    ax.set_xlabel("frames")
                    ax.set_ylabel("fbank coeff")
                    fig.tight_layout()
                else:
                    fig = _plot_and_save_attention(att_w, filename)
                savefn(fig, filename)


class Transformer(TTSInterface, torch.nn.Module):
    """Text-to-Speech Transformer

    Reference:
        Neural Speech Synthesis with Transformer Network
        (https://arxiv.org/pdf/1809.08895.pdf)

    :param int idim: dimension of the inputs
    :param int odim: dimension of the outputs
    :param Namespace args: argments containing following attributes
        (int) embed_dim: dimension of character embedding
        (int) eprenet_conv_layers: number of encoder prenet convolution layers
        (int) eprenet_conv_chans: number of encoder prenet convolution channels
        (int) eprenet_conv_filts: filter size of encoder prenet convolution
        (int) dprenet_layers: number of decoder prenet layers
        (int) dprenet_units: number of decoder prenet hidden units
        (int) elayers: number of encoder layers
        (int) eunits: number of encoder hidden units
        (int) adim: number of attention transformation dimensions
        (int) aheads: number of heads for multi head attention
        (int) dlayers: number of decoder layers
        (int) dunits: number of decoder hidden units
        (int) postnet_layers: number of postnet layers
        (int) postnet_chans: number of postnet channels
        (int) postnet_filts: filter size of postnet
        (bool) use_scaled_pos_enc: whether to use trainable scaled positional encoding instead of the fixed scale one
        (bool) use_batch_norm: whether to use batch normalization in encoder prenet
        (bool) encoder_normalize_before: whether to perform layer normalization before encoder block
        (bool) decoder_normalize_before: whether to perform layer normalization before decoder block
        (bool) encoder_concat_after: whether to concatenate attention layer's input and output in encoder
        (bool) decoder_concat_after: whether to concatenate attention layer's input and output in decoder
        (int) reduction_factor: reduction factor
        (float) transformer_init: how to initialize transformer parameters
        (float) transformer_lr: initial value of learning rate
        (int) transformer_warmup_steps: optimizer warmup steps
        (float) transformer_enc_dropout_rate: dropout rate in encoder except for attention and positional encoding
        (float) transformer_enc_positional_dropout_rate: dropout rate after encoder positional encoding
        (float) transformer_enc_attn_dropout_rate: dropout rate in encoder self-attention module
        (float) transformer_dec_dropout_rate: dropout rate in decoder except for attention and positional encoding
        (float) transformer_dec_positional_dropout_rate:  dropout rate after decoder positional encoding
        (float) transformer_dec_attn_dropout_rate: dropout rate in deocoder self-attention module
        (float) transformer_enc_dec_attn_dropout_rate: dropout rate in encoder-deocoder attention module
        (float) eprenet_dropout_rate: dropout rate in encoder prenet
        (float) dprenet_dropout_rate: dropout rate in decoder prenet
        (float) postnet_dropout_rate: dropout rate in postnet
        (bool) use_masking: whether to use masking in calculation of loss
        (float) bce_pos_weight: positive sample weight in bce calculation (only for use_masking=true)
        (str) loss_type: how to calculate loss
        (bool) use_guided_attn_loss: whether to use guided attention loss
        (int) num_heads_applied_guided_attn: number of heads in each layer to be applied guided attention loss
        (int) num_layers_applied_guided_attn: number of layers to be applied guided attention loss
        (list) modules_applied_guided_attn: list of module names to be applied guided attention loss
    """

    @staticmethod
    def add_arguments(parser):
        group = parser.add_argument_group("transformer model setting")
        # network structure related
        group.add_argument("--embed-dim", default=512, type=int,
                           help="Dimension of character embedding in encoder prenet")
        group.add_argument("--eprenet-conv-layers", default=3, type=int,
                           help="Number of encoder prenet convolution layers")
        group.add_argument("--eprenet-conv-chans", default=256, type=int,
                           help="Number of encoder prenet convolution channels")
        group.add_argument("--eprenet-conv-filts", default=5, type=int,
                           help="Filter size of encoder prenet convolution")
        group.add_argument("--dprenet-layers", default=2, type=int,
                           help="Number of decoder prenet layers")
        group.add_argument("--dprenet-units", default=256, type=int,
                           help="Number of decoder prenet hidden units")
        group.add_argument("--elayers", default=3, type=int,
                           help="Number of encoder layers")
        group.add_argument("--eunits", default=1536, type=int,
                           help="Number of encoder hidden units")
        group.add_argument("--adim", default=384, type=int,
                           help="Number of attention transformation dimensions")
        group.add_argument("--aheads", default=4, type=int,
                           help="Number of heads for multi head attention")
        group.add_argument("--dlayers", default=3, type=int,
                           help="Number of decoder layers")
        group.add_argument("--dunits", default=1536, type=int,
                           help="Number of decoder hidden units")
        group.add_argument("--postnet-layers", default=5, type=int,
                           help="Number of postnet layers")
        group.add_argument("--postnet-chans", default=256, type=int,
                           help="Number of postnet channels")
        group.add_argument("--postnet-filts", default=5, type=int,
                           help="Filter size of postnet")
        group.add_argument("--use-scaled-pos-enc", default=True, type=strtobool,
                           help="Use trainable scaled positional encoding instead of the fixed scale one.")
        group.add_argument("--use-batch-norm", default=True, type=strtobool,
                           help="Whether to use batch normalization")
        group.add_argument("--encoder-normalize-before", default=False, type=strtobool,
                           help="Whether to apply layer norm before encoder block")
        group.add_argument("--decoder-normalize-before", default=False, type=strtobool,
                           help="Whether to apply layer norm before decoder block")
        group.add_argument("--encoder-concat-after", default=False, type=strtobool,
                           help="Whether to concatenate attention layer's input and output in encoder")
        group.add_argument("--decoder-concat-after", default=False, type=strtobool,
                           help="Whether to concatenate attention layer's input and output in decoder")
        parser.add_argument("--reduction-factor", default=1, type=int,
                            help="Reduction factor")
        # training related
        group.add_argument("--transformer-init", type=str, default="pytorch",
                           choices=["pytorch", "xavier_uniform", "xavier_normal",
                                    "kaiming_uniform", "kaiming_normal"],
                           help="How to initialize transformer parameters")
        group.add_argument("--initial-encoder-alpha", type=float, default=1.0,
                           help="Initial alpha value in encoder's ScaledPositionalEncoding")
        group.add_argument("--initial-decoder-alpha", type=float, default=1.0,
                           help="Initial alpha value in decoder's ScaledPositionalEncoding")
        group.add_argument("--transformer-lr", default=1.0, type=float,
                           help="Initial value of learning rate")
        group.add_argument("--transformer-warmup-steps", default=4000, type=int,
                           help="Optimizer warmup steps")
        group.add_argument("--transformer-enc-dropout-rate", default=0.1, type=float,
                           help="Dropout rate for transformer encoder except for attention")
        group.add_argument("--transformer-enc-positional-dropout-rate", default=0.1, type=float,
                           help="Dropout rate for transformer encoder positional encoding")
        group.add_argument("--transformer-enc-attn-dropout-rate", default=0.1, type=float,
                           help="Dropout rate for transformer encoder self-attention")
        group.add_argument("--transformer-dec-dropout-rate", default=0.1, type=float,
                           help="Dropout rate for transformer decoder except for attention and pos encoding")
        group.add_argument("--transformer-dec-positional-dropout-rate", default=0.1, type=float,
                           help="Dropout rate for transformer decoder positional encoding")
        group.add_argument("--transformer-dec-attn-dropout-rate", default=0.1, type=float,
                           help="Dropout rate for transformer decoder self-attention")
        group.add_argument("--transformer-enc-dec-attn-dropout-rate", default=0.1, type=float,
                           help="Dropout rate for transformer encoder-decoder attention")
        group.add_argument("--eprenet-dropout-rate", default=0.5, type=float,
                           help="Dropout rate in encoder prenet")
        group.add_argument("--dprenet-dropout-rate", default=0.5, type=float,
                           help="Dropout rate in decoder prenet")
        group.add_argument("--postnet-dropout-rate", default=0.5, type=float,
                           help="Dropout rate in postnet")
        # loss related
        group.add_argument("--use-masking", default=True, type=strtobool,
                           help="Whether to use masking in calculation of loss")
        group.add_argument("--loss-type", default="L1", choices=["L1", "L2", "L1+L2"],
                           help="How to calc loss")
        group.add_argument("--bce-pos-weight", default=5.0, type=float,
                           help="Positive sample weight in BCE calculation (only for use-masking=True)")
        group.add_argument("--use-guided-attn-loss", default=False, type=strtobool,
                           help="Whether to use guided attention loss")
        group.add_argument("--guided-attn-loss-sigma", default=0.4, type=float,
                           help="Sigma in guided attention loss")
        group.add_argument("--num-heads-applied-guided-attn", default=2, type=int,
                           help="Number of heads in each layer to be applied guided attention loss"
                                "if set -1, all of the heads will be applied.")
        group.add_argument("--num-layers-applied-guided-attn", default=2, type=int,
                           help="Number of layers to be applied guided attention loss"
                                "if set -1, all of the layers will be applied.")
        group.add_argument("--modules-applied-guided-attn", type=str, nargs="+",
                           default=["encoder-decoder"],
                           help="Module name list to be applied guided attention loss")
        return parser

    @property
    def attention_plot_class(self):
        return TTSPlot

    def __init__(self, idim, odim, args):
        # initialize base classes
        TTSInterface.__init__(self)
        torch.nn.Module.__init__(self)

        # store hyperparameters
        self.idim = idim
        self.odim = odim
        self.use_scaled_pos_enc = args.use_scaled_pos_enc
        self.reduction_factor = args.reduction_factor
        self.loss_type = args.loss_type
        self.use_guided_attn_loss = args.use_guided_attn_loss
        if self.use_guided_attn_loss:
            if args.num_layers_applied_guided_attn == -1:
                self.num_layers_applied_guided_attn = args.elayers
            else:
                self.num_layers_applied_guided_attn = args.num_layers_applied_guided_attn
            if args.num_heads_applied_guided_attn == -1:
                self.num_heads_applied_guided_attn = args.aheads
            else:
                self.num_heads_applied_guided_attn = args.num_heads_applied_guided_attn
            self.modules_applied_guided_attn = args.modules_applied_guided_attn

        # use idx 0 as padding idx
        padding_idx = 0

        # get positional encoding class
        pos_enc_class = ScaledPositionalEncoding if self.use_scaled_pos_enc else PositionalEncoding

        # define transformer encoder
        if args.eprenet_conv_layers != 0:
            # encoder prenet
            encoder_input_layer = torch.nn.Sequential(
                EncoderPrenet(
                    idim=idim,
                    embed_dim=args.embed_dim,
                    elayers=0,
                    econv_layers=args.eprenet_conv_layers,
                    econv_chans=args.eprenet_conv_chans,
                    econv_filts=args.eprenet_conv_filts,
                    use_batch_norm=args.use_batch_norm,
                    dropout_rate=args.eprenet_dropout_rate,
                    padding_idx=padding_idx
                ),
                torch.nn.Linear(args.eprenet_conv_chans, args.adim)
            )
        else:
            encoder_input_layer = torch.nn.Embedding(
                num_embeddings=idim,
                embedding_dim=args.adim,
                padding_idx=padding_idx
            )
        self.encoder = Encoder(
            idim=idim,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            linear_units=args.eunits,
            num_blocks=args.elayers,
            input_layer=encoder_input_layer,
            dropout_rate=args.transformer_enc_dropout_rate,
            positional_dropout_rate=args.transformer_enc_positional_dropout_rate,
            attention_dropout_rate=args.transformer_enc_attn_dropout_rate,
            pos_enc_class=pos_enc_class,
            normalize_before=args.encoder_normalize_before,
            concat_after=args.encoder_concat_after
        )

        # define transformer decoder
        if args.dprenet_layers != 0:
            # decoder prenet
            decoder_input_layer = torch.nn.Sequential(
                DecoderPrenet(
                    idim=odim,
                    n_layers=args.dprenet_layers,
                    n_units=args.dprenet_units,
                    dropout_rate=args.dprenet_dropout_rate
                ),
                torch.nn.Linear(args.dprenet_units, args.adim)
            )
        else:
            decoder_input_layer = "linear"
        self.decoder = Decoder(
            odim=-1,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            linear_units=args.dunits,
            num_blocks=args.dlayers,
            dropout_rate=args.transformer_dec_dropout_rate,
            positional_dropout_rate=args.transformer_dec_positional_dropout_rate,
            self_attention_dropout_rate=args.transformer_dec_attn_dropout_rate,
            src_attention_dropout_rate=args.transformer_enc_dec_attn_dropout_rate,
            input_layer=decoder_input_layer,
            use_output_layer=False,
            pos_enc_class=pos_enc_class,
            normalize_before=args.decoder_normalize_before,
            concat_after=args.decoder_concat_after
        )

        # define final projection
        self.feat_out = torch.nn.Linear(args.adim, odim * args.reduction_factor)
        self.prob_out = torch.nn.Linear(args.adim, args.reduction_factor)

        # define postnet
        self.postnet = None if args.postnet_layers == 0 else Postnet(
            idim=idim,
            odim=odim,
            n_layers=args.postnet_layers,
            n_chans=args.postnet_chans,
            n_filts=args.postnet_filts,
            use_batch_norm=args.use_batch_norm,
            dropout_rate=args.postnet_dropout_rate
        )

        # define loss function
        self.criterion = TransformerLoss(args)
        if self.use_guided_attn_loss:
            self.attn_criterion = GuidedMultiHeadAttentionLoss(args.guided_attn_loss_sigma)

        # initialize parameters
        self._reset_parameters(args)

    def _reset_parameters(self, args):
        if self.use_scaled_pos_enc:
            # alpha in scaled positional encoding init
            self.encoder.embed[-1].alpha.data = torch.tensor(args.initial_encoder_alpha)
            self.decoder.embed[-1].alpha.data = torch.tensor(args.initial_decoder_alpha)

        if args.transformer_init == "pytorch":
            return
        # weight init
        for p in self.parameters():
            if p.dim() > 1:
                if args.transformer_init == "xavier_uniform":
                    torch.nn.init.xavier_uniform_(p.data)
                elif args.transformer_init == "xavier_normal":
                    torch.nn.init.xavier_normal_(p.data)
                elif args.transformer_init == "kaiming_uniform":
                    torch.nn.init.kaiming_uniform_(p.data, nonlinearity="relu")
                elif args.transformer_init == "kaiming_normal":
                    torch.nn.init.kaiming_normal_(p.data, nonlinearity="relu")
                else:
                    raise ValueError("Unknown initialization: " + args.transformer_init)
        # bias init
        for p in self.parameters():
            if p.dim() == 1:
                p.data.zero_()
        # reset some modules with default init
        for m in self.modules():
            if isinstance(m, (torch.nn.Embedding, LayerNorm)):
                m.reset_parameters()

    def _add_first_frame_and_remove_last_frame(self, ys):
        ys_in = torch.cat([ys.new_zeros((ys.shape[0], 1, ys.shape[2])), ys[:, :-1]], dim=1)
        return ys_in

    def forward(self, xs, ilens, ys, labels, olens, *args, **kwargs):
        """Calculate forward propagation

        :param torch.Tensor xs: batch of padded character ids (B, Tmax)
        :param torch.Tensor ilens: list of lengths of each input batch (B)
        :param torch.Tensor ys: batch of padded target features (B, Lmax, odim)
        :param torch.Tensor olens: batch of the lengths of each target (B)
        :return: loss value
        :rtype: torch.Tensor
        """
        # remove unnecessary padded part (for multi-gpus)
        max_ilen = max(ilens)
        max_olen = max(olens)
        if max_ilen != xs.shape[1]:
            xs = xs[:, :max_ilen]
        if max_olen != ys.shape[1]:
            ys = ys[:, :max_olen]
            labels = labels[:, :max_olen]

        # forward encoder
        x_masks = self._source_mask(ilens)
        hs, _ = self.encoder(xs, x_masks)

        # thin out frames for reduction factor (B, Lmax, odim) ->  (B, Lmax//r, odim)
        if self.reduction_factor > 1:
            ys_in = ys[:, self.reduction_factor - 1::self.reduction_factor]
            olens_in = olens.new([olen // self.reduction_factor for olen in olens])
        else:
            ys_in, olens_in = ys, olens

        # add first zero frame and remove last frame for auto-regressive
        ys_in = self._add_first_frame_and_remove_last_frame(ys_in)

        # forward decoder
        y_masks = self._target_mask(olens_in)
        xy_masks = self._source_to_target_mask(ilens, olens_in)
        zs, _ = self.decoder(ys_in, y_masks, hs, xy_masks)
        # (B, Lmax//r, odim * r) -> (B, Lmax//r * r, odim)
        before_outs = self.feat_out(zs).view(zs.size(0), -1, self.odim)
        # (B, Lmax//r, r) -> (B, Lmax//r * r)
        logits = self.prob_out(zs).view(zs.size(0), -1)

        # postnet -> (B, Lmax//r * r, odim)
        if self.postnet is None:
            after_outs = before_outs
        else:
            after_outs = before_outs + self.postnet(before_outs.transpose(1, 2)).transpose(1, 2)

        # modifiy mod part of groundtruth
        if self.reduction_factor > 1:
            olens = olens.new([olen - olen % self.reduction_factor for olen in olens])
            max_olen = max(olens)
            ys = ys[:, :max_olen]
            labels = labels[:, :max_olen]
            labels[:, -1] = 1.0  # make sure at least one frame has 1

        # caluculate loss values
        l1_loss, l2_loss, bce_loss = self.criterion(
            after_outs, before_outs, logits, ys, labels, olens)
        if self.loss_type == "L1":
            loss = l1_loss + bce_loss
        elif self.loss_type == "L2":
            loss = l2_loss + bce_loss
        elif self.loss_type == "L1+L2":
            loss = l1_loss + l2_loss + bce_loss
        else:
            raise ValueError("unknown --loss-type " + self.loss_type)
        report_keys = [
            {"l1_loss": l1_loss.item()},
            {"l2_loss": l2_loss.item()},
            {"bce_loss": bce_loss.item()},
            {"loss": loss.item()},
        ]

        # calculate guided attention loss
        if self.use_guided_attn_loss:
            # calculate for encoder
            if "encoder" in self.modules_applied_guided_attn:
                att_ws = []
                for idx, layer_idx in enumerate(reversed(range(len(self.encoder.encoders)))):
                    att_ws += [self.encoder.encoders[layer_idx].self_attn.attn[:, :self.num_heads_applied_guided_attn]]
                    if idx + 1 == self.num_layers_applied_guided_attn:
                        break
                att_ws = torch.cat(att_ws, dim=1)  # (B, H*L, T_in, T_in)
                enc_attn_loss = self.attn_criterion(att_ws, ilens, ilens)
                loss = loss + enc_attn_loss
                report_keys += [{"enc_attn_loss": enc_attn_loss.item()}]
            # calculate for decoder
            if "decoder" in self.modules_applied_guided_attn:
                att_ws = []
                for idx, layer_idx in enumerate(reversed(range(len(self.decoder.decoders)))):
                    att_ws += [self.decoder.decoders[layer_idx].self_attn.attn[:, :self.num_heads_applied_guided_attn]]
                    if idx + 1 == self.num_layers_applied_guided_attn:
                        break
                att_ws = torch.cat(att_ws, dim=1)  # (B, H*L, T_out, T_out)
                dec_attn_loss = self.attn_criterion(att_ws, olens_in, olens_in)
                loss = loss + dec_attn_loss
                report_keys += [{"dec_attn_loss": dec_attn_loss.item()}]
            # calculate for encoder-decoder
            if "encoder-decoder" in self.modules_applied_guided_attn:
                att_ws = []
                for idx, layer_idx in enumerate(reversed(range(len(self.decoder.decoders)))):
                    att_ws += [self.decoder.decoders[layer_idx].src_attn.attn[:, :self.num_heads_applied_guided_attn]]
                    if idx + 1 == self.num_layers_applied_guided_attn:
                        break
                att_ws = torch.cat(att_ws, dim=1)  # (B, H*L, T_out, T_in)
                enc_dec_attn_loss = self.attn_criterion(att_ws, ilens, olens_in)
                loss = loss + enc_dec_attn_loss
                report_keys += [{"enc_dec_attn_loss": enc_dec_attn_loss.item()}]

        # report extra information
        if self.use_scaled_pos_enc:
            report_keys += [
                {"encoder_alpha": self.encoder.embed[-1].alpha.data.item()},
                {"decoder_alpha": self.decoder.embed[-1].alpha.data.item()},
            ]
        self.reporter.report(report_keys)

        return loss

    def inference(self, x, inference_args, *args, **kwargs):
        """Generates the sequence of features from given a sequences of characters

        :param torch.Tensor x: the sequence of character ids (T)
        :param Namespace inference_args: argments containing following attributes
            (float) threshold: threshold in inference
            (float) minlenratio: minimum length ratio in inference
            (float) maxlenratio: maximum length ratio in inference
        :rtype: torch.Tensor
        :return: the sequence of stop probabilities (L)
        :rtype: torch.Tensor
        """
        # get options
        threshold = inference_args.threshold
        minlenratio = inference_args.minlenratio
        maxlenratio = inference_args.maxlenratio

        # forward encoder
        xs = x.unsqueeze(0)
        hs, _ = self.encoder(xs, None)

        # set limits of length
        maxlen = int(hs.size(1) * maxlenratio / self.reduction_factor)
        minlen = int(hs.size(1) * minlenratio / self.reduction_factor)

        # initialize
        idx = 0
        ys = hs.new_zeros(1, 1, self.odim)
        outs, probs = [], []

        # forward decoder step-by-step
        while True:
            # update index
            idx += 1

            # calculate output and stop prob at idx-th step
            y_masks = subsequent_mask(idx).unsqueeze(0)
            z = self.decoder.recognize(ys, y_masks, hs)  # (B, adim)
            outs += [self.feat_out(z).view(self.reduction_factor, self.odim)]  # [(r, odim), ...]
            probs += [torch.sigmoid(self.prob_out(z))[0]]  # [(r), ...]

            # update next inputs
            ys = torch.cat((ys, outs[-1][-1].view(1, 1, self.odim)), dim=1)  # (1, idx + 1, odim)

            # check whether to finish generation
            if int(sum(probs[-1] >= threshold)) > 0 or idx >= maxlen:
                # check mininum length
                if idx < minlen:
                    continue
                outs = torch.cat(outs, dim=0).unsqueeze(0).transpose(1, 2)  # (L, odim) -> (1, L, odim) -> (1, odim, L)
                if self.postnet is not None:
                    outs = outs + self.postnet(outs)  # (1, odim, L)
                outs = outs.transpose(2, 1).squeeze(0)  # (L, odim)
                probs = torch.cat(probs, dim=0)
                break

        return outs, probs

    def calculate_all_attentions(self, xs, ilens, ys, olens, *args, **kwargs):
        """Calculate attention weights of all of the layers

        :param torch.Tensor xs: batch of padded character ids (B, Tmax)
        :param torch.Tensor ilens: list of lengths of each input batch (B)
        :param torch.Tensor ys: batch of padded target features (B, Lmax, odim)
        :param torch.Tensor ilens: list of lengths of each output batch (B)
        :return: attention weights dict
        :rtype: dict
        """
        with torch.no_grad():
            # forward encoder
            x_masks = self._source_mask(ilens)
            hs, _ = self.encoder(xs, x_masks)

            # thin out frames for reduction factor (B, Lmax, odim) ->  (B, Lmax//r, odim)
            if self.reduction_factor > 1:
                ys_in = ys[:, self.reduction_factor - 1::self.reduction_factor]
                olens_in = olens.new([olen // self.reduction_factor for olen in olens])
            else:
                ys_in, olens_in = ys, olens

            # add first zero frame and remove last frame for auto-regressive
            ys_in = self._add_first_frame_and_remove_last_frame(ys_in)

            # forward decoder
            y_masks = self._target_mask(olens_in)
            xy_masks = self._source_to_target_mask(ilens, olens_in)
            zs, _ = self.decoder(ys_in, y_masks, hs, xy_masks)
            # (B, Lmax//r, odim * r) -> (B, Lmax//r * r, odim)
            before_outs = self.feat_out(zs).view(zs.size(0), -1, self.odim)
            # postnet -> (B, Lmax//r * r, odim)
            if self.postnet is None:
                after_outs = before_outs
            else:
                after_outs = before_outs + self.postnet(before_outs.transpose(1, 2)).transpose(1, 2)

        # modifiy mod part of groundtruth
        if self.reduction_factor > 1:
            olens = olens.new([olen - olen % self.reduction_factor for olen in olens])

        att_ws_dict = dict()
        for name, m in self.named_modules():
            if isinstance(m, MultiHeadedAttention):
                attn = m.attn.cpu().numpy()
                if "encoder" in name:
                    attn = [a[:, :l, :l] for a, l in zip(attn, ilens.tolist())]
                elif "decoder" in name:
                    if "src" in name:
                        attn = [a[:, :ol, :il] for a, il, ol in zip(attn, ilens.tolist(), olens_in.tolist())]
                    elif "self" in name:
                        attn = [a[:, :l, :l] for a, l in zip(attn, olens_in.tolist())]
                    else:
                        logging.warning("unknown attention module: " + name)
                else:
                    logging.warning("unknown attention module: " + name)
                att_ws_dict[name] = attn

        att_ws_dict["before_postnet_fbank"] = [m[:l].T for m, l in zip(before_outs.cpu().numpy(), olens.tolist())]
        att_ws_dict["after_postnet_fbank"] = [m[:l].T for m, l in zip(after_outs.cpu().numpy(), olens.tolist())]
        return att_ws_dict

    def _source_mask(self, ilens):
        """Make mask for MultiHeadedAttention using padded sequences

        >>> ilens = [5, 3]
        >>> self._source_mask(ilens)
        tensor([[[1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1]],

                [[1, 1, 1, 0, 0],
                 [1, 1, 1, 0, 0],
                 [1, 1, 1, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0]]], dtype=torch.uint8)
        """
        x_masks = make_non_pad_mask(ilens).to(next(self.parameters()).device)
        return x_masks.unsqueeze(-2) & x_masks.unsqueeze(-1)

    def _target_mask(self, olens):
        """Make mask for MaskedMultiHeadedAttention using padded sequences

        >>> olens = [5, 3]
        >>> self._target_mask(olens)
        tensor([[[1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0],
                 [1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1]],

                [[1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0]]], dtype=torch.uint8)
        """
        y_masks = make_non_pad_mask(olens).to(next(self.parameters()).device)
        s_masks = subsequent_mask(y_masks.size(-1), device=y_masks.device).unsqueeze(0)
        return y_masks.unsqueeze(-2) & s_masks & y_masks.unsqueeze(-1)

    def _source_to_target_mask(self, ilens, olens):
        """Make source to target mask for MultiHeadedAttention using padded sequences

        >>> ilens = [4, 2]
        >>> olens = [5, 3]
        >>> self._source_to_target_mask(ilens)
        tensor([[[1, 1, 1, 1],
                 [1, 1, 1, 1],
                 [1, 1, 1, 1],
                 [1, 1, 1, 1],
                 [1, 1, 1, 1]],

                [[1, 1, 0, 0],
                 [1, 1, 0, 0],
                 [1, 1, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0]]], dtype=torch.uint8)
        """
        x_masks = make_non_pad_mask(ilens).to(next(self.parameters()).device)
        y_masks = make_non_pad_mask(olens).to(next(self.parameters()).device)
        return x_masks.unsqueeze(-2) & y_masks.unsqueeze(-1)

    @property
    def base_plot_keys(self):
        """base key names to plot during training. keys should match what `chainer.reporter` reports

        if you add the key `loss`, the reporter will report `main/loss` and `validation/main/loss` values.
        also `loss.png` will be created as a figure visulizing `main/loss` and `validation/main/loss` values.

        :rtype list[str] plot_keys: base keys to plot during training
        """
        plot_keys = ["loss", "l1_loss", "l2_loss", "bce_loss"]
        if self.use_scaled_pos_enc:
            plot_keys += ["encoder_alpha", "decoder_alpha"]
        if self.use_guided_attn_loss:
            if "encoder" in self.modules_applied_guided_attn:
                plot_keys += ["enc_attn_loss"]
            if "decoder" in self.modules_applied_guided_attn:
                plot_keys += ["dec_attn_loss"]
            if "encoder-decoder" in self.modules_applied_guided_attn:
                plot_keys += ["enc_dec_attn_loss"]

        return plot_keys

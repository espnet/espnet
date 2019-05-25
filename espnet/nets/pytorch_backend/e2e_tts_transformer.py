# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita and Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


import numpy as np
import torch
import torch.nn.functional as F

from espnet.nets.pytorch_backend.e2e_asr_transformer import subsequent_mask
from espnet.nets.pytorch_backend.e2e_tts_tacotron2 import make_non_pad_mask
from espnet.nets.pytorch_backend.e2e_tts_tacotron2 import Tacotron2Loss
from espnet.nets.pytorch_backend.tacotron2.decoder import Postnet
from espnet.nets.pytorch_backend.tacotron2.decoder import Prenet as DecoderPrenet
from espnet.nets.pytorch_backend.tacotron2.encoder import Encoder as EncoderPrenet
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.decoder import Decoder
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.embedding import ScaledPositionalEncoding
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.plot import PlotAttentionReport
from espnet.nets.tts_interface import TTSInterface
from espnet.utils.cli_utils import strtobool


class TransformerTTSLoss(torch.nn.Module):
    """Transformer TTS loss function

    :param Namespace args: argments containing following attributes
        (bool) use_masking: whether to mask padded part in loss calculation
        (float) bce_pos_weight: weight of positive sample of stop token (only for use_masking=True)
    """

    def __init__(self, args):
        super(TransformerTTSLoss, self).__init__()
        self.use_masking = args.use_masking
        self.bce_pos_weight = args.bce_pos_weight

    def forward(self, outs, logits, ys, labels, olens):
        """Transformer loss forward computation

        :param torch.Tensor outs: outputs i.e. log-mels (B, Lmax, odim)
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
        # prepare weight of positive samples in cross entorpy
        if self.bce_pos_weight != 1.0:
            weights = ys.new(*labels.size()).fill_(1)
            weights.masked_fill_(labels.eq(1), self.bce_pos_weight)
        else:
            weights = None

        # perform masking for padded values
        if self.use_masking:
            mask = make_non_pad_mask(olens).unsqueeze(-1).to(ys.device)
            ys = ys.masked_select(mask)
            outs = outs.masked_select(mask)
            labels = labels.masked_select(mask[:, :, 0])
            logits = logits.masked_select(mask[:, :, 0])
            weights = weights.masked_select(mask[:, :, 0]) if weights is not None else None

        # calculate loss
        l1_loss = F.l1_loss(outs, ys)
        mse_loss = F.mse_loss(outs, ys)
        bce_loss = F.binary_cross_entropy_with_logits(logits, labels, weights)

        return l1_loss, mse_loss, bce_loss


class Transformer(TTSInterface, torch.nn.Module):
    """Transformer for TTS

    Reference: Neural Speech Synthesis with Transformer Network (https://arxiv.org/pdf/1809.08895.pdf)
    """

    @staticmethod
    def add_arguments(parser):
        group = parser.add_argument_group("transformer model setting")
        # network structure related
        group.add_argument('--eprenet-conv-layers', default=3, type=int,
                           help='Number of encoder prenet convolution layers')
        group.add_argument('--eprenet-conv-chans', default=512, type=int,
                           help='Number of encoder prenet convolution channels')
        group.add_argument('--eprenet-conv-filts', default=5, type=int,
                           help='Filter size of encoder prenet convolution')
        group.add_argument('--dprenet-layers', default=2, type=int,
                           help='Number of decoder prenet layers')
        group.add_argument('--dprenet-units', default=256, type=int,
                           help='Number of decoder prenet hidden units')
        group.add_argument('--elayers', default=12, type=int,
                           help='Number of encoder layers')
        group.add_argument('--eunits', default=2048, type=int,
                           help='Number of encoder hidden units')
        group.add_argument('--adim', default=256, type=int,
                           help='Number of attention transformation dimensions')
        group.add_argument('--aheads', default=4, type=int,
                           help='Number of heads for multi head attention')
        group.add_argument('--dlayers', default=6, type=int,
                           help='Number of decoder layers')
        group.add_argument('--dunits', default=2048, type=int,
                           help='Number of decoder hidden units')
        group.add_argument('--postnet-layers', default=5, type=int,
                           help='Number of postnet layers')
        group.add_argument('--postnet-chans', default=512, type=int,
                           help='Number of postnet channels')
        group.add_argument('--postnet-filts', default=5, type=int,
                           help='Filter size of postnet')
        group.add_argument('--use-scaled-pos-enc', default=True, type=strtobool,
                           help='use trainable scaled positional encoding instead of the fixed scale one.')
        group.add_argument('--use-batch-norm', default=True, type=strtobool,
                           help='Whether to use batch normalization')
        # training related
        group.add_argument("--transformer-init", type=str, default="pytorch",
                           choices=["pytorch", "xavier_uniform", "xavier_normal",
                                    "kaiming_uniform", "kaiming_normal"],
                           help='how to initialize transformer parameters')
        group.add_argument('--transformer-lr', default=10.0, type=float,
                           help='Initial value of learning rate')
        group.add_argument('--transformer-warmup-steps', default=25000, type=int,
                           help='optimizer warmup steps')
        group.add_argument('--transformer-attn-dropout-rate', default=0.0, type=float,
                           help='dropout in transformer attention. use --dropout if None is set')
        group.add_argument('--eprenet-dropout-rate', default=0.5, type=float,
                           help='dropout rate in encoder prenet. use --dropout if None is set')
        group.add_argument('--dprenet-dropout-rate', default=0.5, type=float,
                           help='dropout rate in decoder prenet. use --dropout if None is set')
        group.add_argument('--postnet-dropout-rate', default=0.5, type=float,
                           help='dropout rate in postnet. use --dropout-rate if None is set')
        group.add_argument('--dropout-rate', default=0.1, type=float,
                           help='dropout rate in the other module')
        # loss related
        group.add_argument('--use-masking', default=True, type=strtobool,
                           help='Whether to use masking in calculation of loss')
        group.add_argument('--bce-pos-weight', default=5.0, type=float,
                           help='Positive sample weight in BCE calculation (only for use-masking=True)')
        return parser

    @property
    def attention_plot_class(self):
        return PlotAttentionReport

    def __init__(self, idim, odim, args):
        # initialize base classes
        TTSInterface.__init__(self)
        torch.nn.Module.__init__(self)

        # store hyperparameters
        self.idim = idim
        self.odim = odim
        self.embed_dim = args.embed_dim
        self.eprenet_conv_layers = args.eprenet_conv_layers
        self.eprenet_conv_filts = args.eprenet_conv_filts
        self.eprenet_conv_chans = args.eprenet_conv_chans
        self.dprenet_layers = args.dprenet_layers
        self.dprenet_units = args.dprenet_units
        self.postnet_layers = args.postnet_layers
        self.postnet_chans = args.postnet_chans
        self.postnet_filts = args.postnet_filts
        self.adim = args.adim
        self.aheads = args.aheads
        self.elayers = args.elayers
        self.eunits = args.eunits
        self.dlayers = args.dlayers
        self.dunits = args.dunits
        self.use_batch_norm = args.use_batch_norm
        self.dropout_rate = args.dropout_rate
        if args.eprenet_dropout_rate is None:
            self.eprenet_dropout_rate = args.dropout_rate
        else:
            self.eprenet_dropout_rate = args.eprenet_dropout_rate
        if args.dprenet_dropout_rate is None:
            self.dprenet_dropout_rate = args.dropout_rate
        else:
            self.dprenet_dropout_rate = args.dprenet_dropout_rate
        if args.postnet_dropout_rate is None:
            self.postnet_dropout_rate = args.dropout_rate
        else:
            self.postnet_dropout_rate = args.postnet_dropout_rate
        if args.transformer_attn_dropout_rate is None:
            self.transformer_attn_dropout_rate = args.dropout_rate
        else:
            self.transformer_attn_dropout_rate = args.transformer_attn_dropout_rate
        self.use_scaled_pos_enc = args.use_scaled_pos_enc
        self.pos_enc_class = ScaledPositionalEncoding if args.use_scaled_pos_enc else PositionalEncoding

        # define transformer encoder
        encoder_prenet = torch.nn.Sequential(
            EncoderPrenet(
                idim=self.idim,
                embed_dim=self.embed_dim,
                elayers=0,
                econv_layers=self.eprenet_conv_layers,
                econv_chans=self.eprenet_conv_chans,
                econv_filts=self.eprenet_conv_filts,
                use_batch_norm=self.use_batch_norm,
                dropout_rate=self.eprenet_dropout_rate
            ),
            torch.nn.Linear(self.eprenet_conv_chans, self.adim)
        )
        self.encoder = Encoder(
            idim=self.idim,
            attention_dim=self.adim,
            attention_heads=self.aheads,
            linear_units=self.eunits,
            num_blocks=self.elayers,
            input_layer=encoder_prenet,
            dropout_rate=self.dropout_rate,
            attention_dropout_rate=self.transformer_attn_dropout_rate,
            pos_enc_class=self.pos_enc_class
        )

        # define transformer decoder
        decoder_prenet = torch.nn.Sequential(
            DecoderPrenet(
                idim=self.odim,
                n_layers=self.dprenet_layers,
                n_units=self.dprenet_units,
                dropout_rate=self.dprenet_dropout_rate
            ),
            torch.nn.Linear(self.dprenet_units, self.adim)
        )
        self.decoder = Decoder(
            odim=-1,
            attention_dim=self.adim,
            attention_heads=self.aheads,
            linear_units=self.dunits,
            num_blocks=self.dlayers,
            dropout_rate=self.dropout_rate,
            attention_dropout_rate=self.transformer_attn_dropout_rate,
            input_layer=decoder_prenet,
            use_output_layer=False,
            pos_enc_class=self.pos_enc_class
        )

        # define final projection
        self.feat_out = torch.nn.Linear(self.adim, self.odim, bias=False)
        self.prob_out = torch.nn.Linear(self.adim, 1)

        # define postnet
        self.postnet = Postnet(
            idim=self.idim,
            odim=self.odim,
            n_layers=self.postnet_layers,
            n_chans=self.postnet_chans,
            n_filts=self.postnet_filts,
            use_batch_norm=self.use_batch_norm,
            dropout_rate=self.postnet_dropout_rate
        )

        # define loss function
        self.criterion = Tacotron2Loss(args)

        # initialize parameters
        self._reset_parameters(args)

    def _reset_parameters(self, args):
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

    def forward(self, xs, ilens, ys, labels, olens, *args, **kwargs):
        """Transformer forward computation

        :param torch.Tensor xs: batch of padded character ids (B, Tmax)
        :param torch.Tensor ilens: list of lengths of each input batch (B)
        :param torch.Tensor ys: batch of padded target features (B, Lmax, odim)
        :param torch.Tensor olens: batch of the lengths of each target (B)
        :return: loss value
        :rtype: torch.Tensor
        """
        # check ilens and olens type (should be list of int)
        if isinstance(ilens, torch.Tensor) or isinstance(ilens, np.ndarray):
            ilens = list(map(int, ilens))
        if isinstance(olens, torch.Tensor) or isinstance(olens, np.ndarray):
            olens = list(map(int, olens))

        # remove unnecessary padded part (for multi-gpus)
        max_in = max(ilens)
        max_out = max(olens)
        if max_in != xs.shape[1]:
            xs = xs[:, :max_in]
        if max_out != ys.shape[1]:
            ys = ys[:, :max_out]
            labels = labels[:, :max_out]

        # forward encoder
        src_masks = make_non_pad_mask(ilens).to(xs.device).unsqueeze(-2)
        hs, h_masks = self.encoder(xs, src_masks)

        # forward decoder
        y_masks = self._target_mask(olens)
        zs, z_masks = self.decoder(ys, y_masks, hs, h_masks)
        before_outs = self.feat_out(zs)  # (B, Lmax, odim)
        logits = self.prob_out(zs).squeeze(-1)  # (B, Lmax)

        # postnet
        after_outs = self.postnet(before_outs.transpose(1, 2)).transpose(1, 2)

        # caluculate taco2 loss
        l1_loss, mse_loss, bce_loss = self.criterion(
            after_outs, before_outs, logits, ys, labels, olens)
        loss = l1_loss + mse_loss + bce_loss

        # report for chainer reporter
        report_keys = [
            {'l1_loss': l1_loss.item()},
            {'mse_loss': mse_loss.item()},
            {'bce_loss': bce_loss.item()},
            {'loss': loss.item()},
        ]
        self.reporter.report(report_keys)

        return loss

    def inference(self, x, inference_args, *args, **kwargs):
        """Generates the sequence of features given the sequences of characters

        :param torch.Tensor x: the sequence of characters (T)
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
        maxlen = int(hs.size(1) * maxlenratio)
        minlen = int(hs.size(1) * minlenratio)

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
            outs += [self.feat_out(z)]  # [(1, odim, 1), ...]
            probs += [torch.sigmoid(self.prob_out(z))[0]]

            # update next inputs
            ys = torch.cat((ys, outs[-1].unsqueeze(0)), dim=1)  # (1, idx + 1, odim)

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
        '''Calculate attention weights

        :param torch.Tensor xs: batch of padded character ids (B, Tmax)
        :param torch.Tensor ilens: list of lengths of each input batch (B)
        :param torch.Tensor ys: batch of padded target features (B, Lmax, odim)
        :param torch.Tensor ilens: list of lengths of each output batch (B)
        :return: attention weights dict
        :rtype: dict
        '''
        with torch.no_grad():
            # forward encoder
            src_masks = make_non_pad_mask(ilens).to(xs.device).unsqueeze(-2)
            hs, h_masks = self.encoder(xs, src_masks)

            # forward decoder
            y_masks = self._target_mask(olens)
            self.decoder(ys, y_masks, hs, h_masks)

        att_ws_dict = dict()
        for name, m in self.named_modules():
            if isinstance(m, MultiHeadedAttention):
                att_ws_dict[name] = m.attn.cpu().numpy()
        return att_ws_dict

    def _target_mask(self, olens):
        y_masks = make_non_pad_mask(olens).to(next(self.parameters()).device)
        s_masks = subsequent_mask(y_masks.size(-1), device=y_masks.device).unsqueeze(0)
        return y_masks.unsqueeze(-2) & s_masks

    @property
    def base_plot_keys(self):
        """base key names to plot during training. keys should match what `chainer.reporter` reports

        if you add the key `loss`, the reporter will report `main/loss` and `validation/main/loss` values.
        also `loss.png` will be created as a figure visulizing `main/loss` and `validation/main/loss` values.

        :rtype list[str] plot_keys: base keys to plot during training
        """
        plot_keys = ['loss', 'l1_loss', 'mse_loss', 'bce_loss']
        return plot_keys

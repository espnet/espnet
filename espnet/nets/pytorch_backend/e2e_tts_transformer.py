# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita and Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
import logging

import numpy as np
import torch
import torch.nn.functional as F

from espnet.nets.pytorch_backend.e2e_asr_transformer import subsequent_mask
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
from espnet.nets.pytorch_backend.transformer.plot import PlotAttentionReport
from espnet.nets.pytorch_backend.transformer.plot import _plot_and_save_attention
from espnet.nets.tts_interface import TTSInterface
from espnet.utils.cli_utils import strtobool


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
        """Transformer loss forward computation

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
    """Transformer for TTS

    - Reference: Neural Speech Synthesis with Transformer Network (https://arxiv.org/pdf/1809.08895.pdf)

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
        (bool) use_batch_norm: whether to use batch normalization')
        (float) transformer_init: how to initialize transformer parameters')
        (float) transformer_lr': initial value of learning rate
        (int) transformer_warmup_steps: optimizer warmup steps
        (float) transformer_attn_dropout_rate: dropout in transformer attention. use dropout if none is set
        (float) eprenet_dropout_rate: dropout rate in encoder prenet. use dropout if none is set
        (float) dprenet_dropout_rate: dropout rate in decoder prenet. use dropout if none is set
        (float) postnet_dropout_rate: dropout rate in postnet. use dropout_rate if none is set
        (float) dropout_rate: dropout rate in the other module
        (bool) use_masking: whether to use masking in calculation of loss
        (float) bce_pos_weight: positive sample weight in bce calculation (only for use_masking=true)
    """

    @staticmethod
    def add_arguments(parser):
        group = parser.add_argument_group("transformer model setting")
        # network structure related
        group.add_argument('--embed-dim', default=512, type=int,
                           help='Dimension of character embedding')
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
        group.add_argument('--encoder-normalize-before', default=True, type=strtobool,
                           help='Whether to apply layer norm before encoder block')
        group.add_argument('--decoder-normalize-before', default=True, type=strtobool,
                           help='Whether to apply layer norm before decoder block')
        group.add_argument('--encoder-concate-after', default=False, type=strtobool,
                           help='Whether to concate attention layer\'s input and output in encoder')
        group.add_argument('--decoder-concate-after', default=False, type=strtobool,
                           help='Whether to concate attention layer\'s input and output in decoder')
        parser.add_argument('--reduction-factor', default=1, type=int,
                            help='Reduction factor')
        # training related
        group.add_argument("--transformer-init", type=str, default="pytorch",
                           choices=["pytorch", "xavier_uniform", "xavier_normal",
                                    "kaiming_uniform", "kaiming_normal"],
                           help='how to initialize transformer parameters')
        group.add_argument("--initial-encoder-alpha", type=float, default=1.0,
                           help='initial alpha value in encoder\'s ScaledPositionalEncoding')
        group.add_argument("--initial-decoder-alpha", type=float, default=1.0,
                           help='initial alpha value in decoder\'s ScaledPositionalEncoding')
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
        group.add_argument('--loss-type', default="L1", choices=["L1", "L2", "L1+L2"],
                           help='How to calc loss')
        group.add_argument('--bce-pos-weight', default=5.0, type=float,
                           help='Positive sample weight in BCE calculation (only for use-masking=True)')
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
        self.reduction_factor = args.reduction_factor
        self.encoder_normalize_before = args.encoder_normalize_before
        self.decoder_normalize_before = args.decoder_normalize_before
        self.encoder_concate_after = args.encoder_concate_after
        self.decoder_concate_after = args.decoder_concate_after

        # define transformer encoder
        if self.eprenet_conv_layers != 0:
            # encoder prenet
            encoder_input_layer = torch.nn.Sequential(
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
        else:
            encoder_input_layer = "embed"
        self.encoder = Encoder(
            idim=self.idim,
            attention_dim=self.adim,
            attention_heads=self.aheads,
            linear_units=self.eunits,
            num_blocks=self.elayers,
            input_layer=encoder_input_layer,
            dropout_rate=self.dropout_rate,
            attention_dropout_rate=self.transformer_attn_dropout_rate,
            pos_enc_class=self.pos_enc_class,
            normalize_before=self.encoder_normalize_before,
            concate_after=self.encoder_concate_after
        )

        # define transformer decoder
        if self.dprenet_layers != 0:
            # decoder prenet
            decoder_input_layer = torch.nn.Sequential(
                DecoderPrenet(
                    idim=self.odim,
                    n_layers=self.dprenet_layers,
                    n_units=self.dprenet_units,
                    dropout_rate=self.dprenet_dropout_rate
                ),
                torch.nn.Linear(self.dprenet_units, self.adim)
            )
        else:
            decoder_input_layer = "linear"
        self.decoder = Decoder(
            odim=-1,
            attention_dim=self.adim,
            attention_heads=self.aheads,
            linear_units=self.dunits,
            num_blocks=self.dlayers,
            dropout_rate=self.dropout_rate,
            attention_dropout_rate=self.transformer_attn_dropout_rate,
            input_layer=decoder_input_layer,
            use_output_layer=False,
            pos_enc_class=self.pos_enc_class,
            normalize_before=self.decoder_normalize_before,
            concate_after=self.decoder_concate_after
        )

        # define final projection
        self.feat_out = torch.nn.Linear(self.adim, self.odim * self.reduction_factor)
        self.prob_out = torch.nn.Linear(self.adim, self.reduction_factor)

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
        self.criterion = TransformerLoss(args)
        self.loss_type = args.loss_type

        # initialize parameters
        self._reset_parameters(args)

    def _reset_parameters(self, args):
        if self.use_scaled_pos_enc:
            # alpha in scaled positional encoding init
            self.encoder.embed[-1].alpha.data = torch.tensor(args.initial_encoder_alpha)
            self.decoder.embed[-1].alpha.data = torch.tensor(args.initial_decoder_alpha)

        if args.transformer_init == "pytorch":
            for m in self.modules():
                if hasattr(m, "reset_parameters"):
                    m.reset_parameters()
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

        # thin out frames for reduction factor (B, Lmax, odim) ->  (B, Lmax//r, odim)
        if self.reduction_factor > 1:
            ys_ = ys[:, self.reduction_factor - 1::self.reduction_factor]
            olens_ = [olen // self.reduction_factor for olen in olens]
        else:
            ys_, olens_ = ys, olens

        # forward decoder
        y_masks = self._target_mask(olens_)
        zs, z_masks = self.decoder(ys_, y_masks, hs, h_masks)
        # (B, Lmax//r * r, odim) -> (B, Lmax//r, odim * r)
        before_outs = self.feat_out(zs).view(zs.size(0), -1, self.odim)
        # (B, Lmax//r, r) -> (B, Lmax//r * r)
        logits = self.prob_out(zs).view(zs.size(0), -1)

        # postnet
        after_outs = before_outs + self.postnet(before_outs.transpose(1, 2)).transpose(1, 2)

        # modifiy mod part of groundtruth
        if self.reduction_factor > 1:
            olens = [olen - olen % self.reduction_factor for olen in olens]
            max_out = max(olens)
            ys = ys[:, :max_out]
            labels = labels[:, :max_out]
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

        # report for chainer reporter
        report_keys = [
            {'l1_loss': l1_loss.item()},
            {'l2_loss': l2_loss.item()},
            {'bce_loss': bce_loss.item()},
            {'loss': loss.item()},
        ]
        if self.use_scaled_pos_enc:
            report_keys += [
                {'encoder_alpha': self.encoder.embed[-1].alpha.data.item()},
                {'decoder_alpha': self.decoder.embed[-1].alpha.data.item()},
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
            zs, zs_mask = self.decoder(ys, y_masks, hs, h_masks)
            # (B, Lmax//r * r, odim) -> (B, Lmax//r, odim * r)
            before_outs = self.feat_out(zs).view(zs.size(0), -1, self.odim)
            # postnet
            after_outs = before_outs + self.postnet(before_outs.transpose(1, 2)).transpose(1, 2)

        att_ws_dict = dict()
        for name, m in self.named_modules():
            if isinstance(m, MultiHeadedAttention):
                attn = m.attn.cpu().numpy()
                if "encoder" in name:
                    attn = [a[:, :l, :l] for a, l in zip(attn, ilens.tolist())]
                elif "decoder" in name:
                    if "src" in name:
                        attn = [a[:, :ol, :il] for a, il, ol in zip(attn, ilens.tolist(), olens.tolist())]
                    elif "self" in name:
                        attn = [a[:, :l, :l] for a, l in zip(attn, olens.tolist())]
                    else:
                        logging.warning("unknown attention module: " + name)
                else:
                    logging.warning("unknown attention module: " + name)
                att_ws_dict[name] = attn

        att_ws_dict["before_postnet_fbank"] = [m[:l].T for m, l in zip(before_outs.cpu().numpy(), olens.tolist())]
        att_ws_dict["after_postnet_fbank"] = [m[:l].T for m, l in zip(after_outs.cpu().numpy(), olens.tolist())]
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
        plot_keys = ['loss', 'l1_loss', 'l2_loss', 'bce_loss']
        if self.use_scaled_pos_enc:
            plot_keys += ['encoder_alpha', 'decoder_alpha']

        return plot_keys

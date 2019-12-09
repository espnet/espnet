"""Self-attention Transducer speech recognition model (pytorch)."""

from distutils.util import strtobool

import logging
import math

import chainer
import torch

from chainer import reporter

from espnet.nets.asr_interface import ASRInterface

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask

from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.transformer.initializer import initialize
from espnet.nets.pytorch_backend.transformer.mask import target_mask
from espnet.nets.pytorch_backend.transformer.plot import PlotAttentionReport

from espnet.nets.pytorch_backend.transducer.loss import TransLoss
from espnet.nets.pytorch_backend.transducer.transformer_decoder import Decoder
from espnet.nets.pytorch_backend.transducer.utils import prepare_loss_inputs


class Reporter(chainer.Chain):
    """A chainer reporter wrapper for transducer-based models."""

    def report(self, loss, cer, wer):
        """Instantiate reporter attributes."""
        reporter.report({'cer': cer}, self)
        reporter.report({'wer': wer}, self)
        logging.info('loss:' + str(loss))
        reporter.report({'loss': loss}, self)


class E2E(ASRInterface, torch.nn.Module):
    """E2E module.

    Args:
        idim (int): dimension of inputs
        odim (int): dimension of outputs
        args (Namespace): argument Namespace containing options

    """

    @staticmethod
    def add_arguments(parser):
        """Extend arguments for self-attention transducer."""
        group = parser.add_argument_group("transformer model setting")

        # Encoder
        group.add_argument('--elayers', default=4, type=int,
                           help='Number of encoder layers (for shared recognition part in multi-speaker asr mode)')
        group.add_argument('--eunits', '-u', default=300, type=int,
                           help='Number of encoder hidden units')
        group.add_argument('--dropout-rate', default=0.0, type=float,
                           help='Dropout rate for the encoder')

        # Attention
        group.add_argument('--adim', default=320, type=int,
                           help='Number of attention transformation dimensions')
        group.add_argument('--aheads', default=4, type=int,
                           help='Number of heads for multi head attention')

        # Decoder
        group.add_argument('--dlayers', default=1, type=int,
                           help='Number of decoder layers')
        group.add_argument('--dunits', default=320, type=int,
                           help='Number of decoder hidden units')
        # Transformer
        group.add_argument("--transformer-init", type=str, default="pytorch",
                           choices=["pytorch", "xavier_uniform", "xavier_normal",
                                    "kaiming_uniform", "kaiming_normal"],
                           help='how to initialize transformer parameters')
        group.add_argument("--transformer-input-layer", type=str, default="conv2d",
                           choices=["conv2d", "linear", "embed"],
                           help='transformer input layer type')
        group.add_argument('--transformer-attn-dropout-rate', default=None, type=float,
                           help='dropout in transformer attention. use --dropout-rate if None is set')
        group.add_argument('--transformer-lr', default=10.0, type=float,
                           help='Initial value of learning rate')
        group.add_argument('--transformer-warmup-steps', default=25000, type=int,
                           help='optimizer warmup steps')
        group.add_argument('--transformer-length-normalized-loss', default=True, type=strtobool,
                           help='normalize loss by length')

        # Transducer
        group.add_argument('--trans-type', default='warp-transducer', type=str,
                           choices=['warp-transducer'],
                           help='Type of transducer implementation to calculate loss.')
        group.add_argument('--joint-dim', default=320, type=int,
                           help='Number of dimensions in joint space')
        group.add_argument('--score-norm-transducer', type=strtobool,
                           nargs='?', default=True,
                           help='Normalize transducer scores by length')

        return parser

    @property
    def attention_plot_class(self):
        """Construct a PlotAttentionReport."""
        return PlotAttentionReport

    def __init__(self, idim, odim, args, ignore_id=-1, asr_model=None, mt_model=None):
        """Construct an E2E object for self-attention transducer model.

        Args:
            idim (int): dimension of inputs
            odim (int): dimension of outputs
            args (Namespace): argument Namespace containing options

        """
        torch.nn.Module.__init__(self)

        if args.transformer_attn_dropout_rate is None:
            args.transformer_attn_dropout_rate = args.dropout_rate

        self.encoder = Encoder(
            idim=idim,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            linear_units=args.eunits,
            num_blocks=args.elayers,
            input_layer=args.transformer_input_layer,
            dropout_rate=args.dropout_rate,
            positional_dropout_rate=args.dropout_rate,
            attention_dropout_rate=args.transformer_attn_dropout_rate)

        self.decoder = Decoder(
            odim=odim,
            jdim=args.joint_dim,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            linear_units=args.dunits,
            num_blocks=args.dlayers,
            dropout_rate=args.dropout_rate,
            positional_dropout_rate=args.dropout_rate,
            self_attention_dropout_rate=args.transformer_attn_dropout_rate)

        self.sos = odim - 1
        self.eos = odim - 1
        self.blank_id = 0
        self.ignore_id = ignore_id

        self.odim = odim
        self.adim = args.adim
        self.subsample = [1]

        self.reporter = Reporter()

        self.criterion = TransLoss(args.trans_type, self.blank_id)

        self.reset_parameters(args)

        self.report_cer = False
        self.report_wer = False

        self.rnnlm = None

        self.reset_parameters(args)

    def reset_parameters(self, args):
        """Initialize/reset parameters."""
        initialize(self, args.transformer_init)

    def forward(self, xs_pad, ilens, ys_pad, ys_pad_asr=None):
        """E2E forward.

        Args:
            xs_pad (torch.Tensor): batch of padded source sequences (B, Tmax, idim)
            ilens (torch.Tensor): batch of lengths of input sequences (B)
            ys_pad (torch.Tensor): batch of padded target sequences (B, Lmax)

        Returns:
            loss (torch.Tensor): self-attention transducer loss value

        """
        # 1. encoder
        xs_pad = xs_pad[:, :max(ilens)]  # for data parallel
        src_mask = (~make_pad_mask(ilens.tolist())).to(xs_pad.device).unsqueeze(-2)
        hs_pad, hs_mask = self.encoder(xs_pad, src_mask)
        self.hs_pad = hs_pad

        # 1.5. transducer preparation related
        ys_in_pad, target, pred_len, target_len = prepare_loss_inputs(ys_pad, hs_mask)
        ys_mask = target_mask(ys_in_pad, self.ignore_id)

        # 2. decoder
        pred_pad, pred_mask = self.decoder(ys_in_pad, ys_mask, hs_pad)
        self.pred_pad = pred_pad

        # 3. loss computation
        loss = self.criterion(pred_pad, target, pred_len, target_len)

        self.loss = loss
        loss_data = float(self.loss)

        # 3. (pending) compute cer/wer
        cer, wer = None, None

        if not math.isnan(loss_data):
            self.reporter.report(loss_data, cer, wer)
        else:
            logging.warning('loss (=%f) is not correct', loss_data)

        return self.loss

    def encode(self, feat):
        """Encode acoustic features."""
        self.eval()
        feat = torch.as_tensor(feat).unsqueeze(0)
        enc_output, _ = self.encoder(feat, None)
        return enc_output.squeeze(0)

    def recognize(self, feat, recog_args, char_list=None, rnnlm=None, use_jit=False):
        """E2E recognize.

        Args:
            x (ndarray): input acoustic feature (T, D)
            recog_args (namespace): argument Namespace containing options
            char_list (list): list of characters
            rnnlm (torch.nn.Module): language model module

        Returns:
            y (list): n-best decoding results

        """
        h = self.encode(feat)
        logging.info('input lengths: ' + str(h.size(0)))

        if recog_args.beam_size == 1:
            nbest_hyps = self.decoder.recognize(h, recog_args, self.blank_id)
        else:
            nbest_hyps = self.decoder.recognize_beam(h, recog_args, self.blank_id, rnnlm)

        return nbest_hyps

    def calculate_all_attentions(self, xs_pad, ilens, ys_pad, ys_pad_asr=None):
        """E2E attention calculation.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, Lmax)
        :return: attention weights with the following shape,
            1) multi-head case => attention weights (B, H, Lmax, Tmax),
            2) other case => attention weights (B, Lmax, Tmax).
        :rtype: float ndarray

        """
        with torch.no_grad():
            self.forward(xs_pad, ilens, ys_pad)
        ret = dict()
        for name, m in self.named_modules():
            if isinstance(m, MultiHeadedAttention):
                ret[name] = m.attn.cpu().numpy()
        return ret

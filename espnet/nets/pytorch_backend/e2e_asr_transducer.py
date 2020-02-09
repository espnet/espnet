"""Transducer speech recognition model (pytorch)."""

from distutils.util import strtobool

import logging
import math

import chainer
import torch

from chainer import reporter

from espnet.nets.asr_interface import ASRInterface


from espnet.nets.pytorch_backend.nets_utils import get_subsample
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.nets_utils import to_device
from espnet.nets.pytorch_backend.nets_utils import to_torch_tensor

from espnet.nets.pytorch_backend.rnn.attentions import att_for
from espnet.nets.pytorch_backend.rnn.encoders import encoder_for

from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.transformer.mask import target_mask

from espnet.nets.pytorch_backend.transducer.initializer import initializer
from espnet.nets.pytorch_backend.transducer.loss import TransLoss
from espnet.nets.pytorch_backend.transducer.rnn_decoders import decoder_for
from espnet.nets.pytorch_backend.transducer.transformer_decoder import Decoder
from espnet.nets.pytorch_backend.transducer.utils import prepare_loss_inputs


class Reporter(chainer.Chain):
    """A chainer reporter wrapper for transducer models."""

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
        """Extend arguments for transducer models.

        Both Transformer and RNN modules are supported.
        General options encapsulate both modules options.

        """
        group = parser.add_argument_group("transformer model setting")

        # Encoder - general
        group.add_argument('--etype', default='blstmp', type=str,
                           choices=['transformer', 'lstm', 'blstm', 'lstmp', 'blstmp',
                                    'vgglstmp', 'vggblstmp', 'vgglstm', 'vggblstm',
                                    'gru', 'bgru', 'grup', 'bgrup',
                                    'vgggrup', 'vggbgrup', 'vgggru', 'vggbgru'],
                           help='Type of encoder network architecture')
        group.add_argument('--elayers', default=4, type=int,
                           help='Number of encoder layers (for shared recognition part in multi-speaker asr mode)')
        group.add_argument('--eunits', '-u', default=300, type=int,
                           help='Number of encoder hidden units')
        group.add_argument('--dropout-rate', default=0.0, type=float,
                           help='Dropout rate for the encoder')
        # Encoder - RNN
        group.add_argument('--eprojs', default=320, type=int,
                           help='Number of encoder projection units')
        group.add_argument('--subsample', default="1", type=str,
                           help='Subsample input frames x_y_z means subsample every x frame \
                           at 1st layer, every y frame at 2nd layer etc.')
        # Attention - general
        group.add_argument('--adim', default=320, type=int,
                           help='Number of attention transformation dimensions')
        group.add_argument('--aheads', default=4, type=int,
                           help='Number of heads for multi head attention')
        group.add_argument('--transformer-attn-dropout-rate', default=0.0, type=float,
                           help='dropout in transformer attention.')
        group.add_argument('--transformer-attn-dropout-rate-encoder', default=0.0, type=float,
                           help='dropout in transformer decoder attention.')
        group.add_argument('--transformer-attn-dropout-rate-decoder', default=0.0, type=float,
                           help='dropout in transformer decoder attention.')
        # Attention - RNN
        group.add_argument('--atype', default='dot', type=str,
                           choices=['noatt', 'dot', 'add', 'location', 'coverage',
                                    'coverage_location', 'location2d', 'location_recurrent',
                                    'multi_head_dot', 'multi_head_add', 'multi_head_loc',
                                    'multi_head_multi_res_loc'],
                           help='Type of attention architecture')
        group.add_argument('--awin', default=5, type=int,
                           help='Window size for location2d attention')
        group.add_argument('--aconv-chans', default=-1, type=int,
                           help='Number of attention convolution channels \
                           (negative value indicates no location-aware attention)')
        group.add_argument('--aconv-filts', default=100, type=int,
                           help='Number of attention convolution filters \
                           (negative value indicates no location-aware attention)')
        # Decoder - general
        group.add_argument('--dtype', default='lstm', type=str,
                           choices=['lstm', 'gru', 'transformer'],
                           help='Type of decoder to use.')
        group.add_argument('--dlayers', default=1, type=int,
                           help='Number of decoder layers')
        group.add_argument('--dunits', default=320, type=int,
                           help='Number of decoder hidden units')
        group.add_argument('--dropout-rate-decoder', default=0.0, type=float,
                           help='Dropout rate for the decoder')
        # Decoder - RNN
        group.add_argument('--dec-embed-dim', default=320, type=int,
                           help='Number of decoder embeddings dimensions')
        group.add_argument('--dropout-rate-embed-decoder', default=0.0, type=float,
                           help='Dropout rate for the decoder embeddings')
        # Transformer
        group.add_argument("--transformer-init", type=str, default="pytorch",
                           choices=["pytorch", "xavier_uniform", "xavier_normal",
                                    "kaiming_uniform", "kaiming_normal"],
                           help='how to initialize transformer parameters')
        group.add_argument("--transformer-input-layer", type=str, default="conv2d",
                           choices=["conv2d", "vgg2l", "linear", "embed"],
                           help='transformer encoder input layer type')
        group.add_argument("--transformer-dec-input-layer", type=str, default="embed",
                           choices=["linear", "embed"],
                           help='transformer decoder input layer type')
        group.add_argument('--transformer-lr', default=10.0, type=float,
                           help='Initial value of learning rate')
        group.add_argument('--transformer-warmup-steps', default=25000, type=int,
                           help='optimizer warmup steps')
        # Transducer
        group.add_argument('--trans-type', default='warp-transducer', type=str,
                           choices=['warp-transducer'],
                           help='Type of transducer implementation to calculate loss.')
        group.add_argument('--rnnt-mode', default='rnnt', type=str,
                           choices=['rnnt', 'rnnt-att'],
                           help='Transducer mode for RNN decoder.')
        group.add_argument('--joint-dim', default=320, type=int,
                           help='Number of dimensions in joint space')
        group.add_argument('--score-norm-transducer', type=strtobool,
                           nargs='?', default=True,
                           help='Normalize transducer scores by length')

        return parser

    def __init__(self, idim, odim, args, ignore_id=-1, blank_id=0):
        """Construct an E2E object for transducer model.

        Args:
            idim (int): dimension of inputs
            odim (int): dimension of outputs
            args (Namespace): argument Namespace containing options

        """
        torch.nn.Module.__init__(self)

        if args.etype == 'transformer':
            self.encoder = Encoder(
                idim=idim,
                attention_dim=args.adim,
                attention_heads=args.aheads,
                linear_units=args.eunits,
                num_blocks=args.elayers,
                input_layer=args.transformer_input_layer,
                dropout_rate=args.dropout_rate,
                positional_dropout_rate=args.dropout_rate,
                attention_dropout_rate=args.transformer_attn_dropout_rate_encoder)

            self.subsample = [1]
        else:
            self.subsample = get_subsample(args, mode='asr', arch='rnn-t')

            self.encoder = encoder_for(args, idim, self.subsample)

        if args.dtype == 'transformer':
            self.decoder = Decoder(
                odim=odim,
                jdim=args.joint_dim,
                attention_dim=args.adim,
                attention_heads=args.aheads,
                linear_units=args.dunits,
                num_blocks=args.dlayers,
                input_layer=args.transformer_dec_input_layer,
                dropout_rate=args.dropout_rate_decoder,
                positional_dropout_rate=args.dropout_rate_decoder,
                attention_dropout_rate=args.transformer_attn_dropout_rate_decoder)
        else:
            if args.etype == 'transformer':
                args.eprojs = args.adim

            if args.rnnt_mode == 'rnnt-att':
                self.att = att_for(args)
                self.decoder = decoder_for(args, odim, self.att)
            else:
                self.decoder = decoder_for(args, odim)

        self.etype = args.etype
        self.dtype = args.dtype
        self.rnnt_mode = args.rnnt_mode

        self.sos = odim - 1
        self.eos = odim - 1
        self.blank_id = blank_id
        self.ignore_id = ignore_id

        self.space = args.sym_space
        self.blank = args.sym_blank

        self.odim = odim
        self.adim = args.adim

        self.reporter = Reporter()

        self.criterion = TransLoss(args.trans_type, self.blank_id)

        self.default_parameters(args)

        if args.report_cer or args.report_wer:
            from espnet.nets.e2e_asr_common import ErrorCalculatorTrans

            self.error_calculator = ErrorCalculatorTrans(self.decoder, args)
        else:
            self.error_calculator = None

        self.logzero = -10000000000.0
        self.loss = None
        self.rnnlm = None

    def default_parameters(self, args):
        """Initialize/reset parameters for transducer."""
        initializer(self, args)

    def forward(self, xs_pad, ilens, ys_pad):
        """E2E forward.

        Args:
            xs_pad (torch.Tensor): batch of padded source sequences (B, Tmax, idim)
            ilens (torch.Tensor): batch of lengths of input sequences (B)
            ys_pad (torch.Tensor): batch of padded target sequences (B, Lmax)

        Returns:
            loss (torch.Tensor): transducer loss value

        """
        # 1. encoder
        if self.etype == 'transformer':
            xs_pad = xs_pad[:, :max(ilens)]
            src_mask = (~make_pad_mask(ilens.tolist())).to(xs_pad.device).unsqueeze(-2)

            hs_pad, hs_mask = self.encoder(xs_pad, src_mask)
        else:
            hs_pad, hlens = xs_pad, ilens
            hs_pad, hlens, _ = self.encoder(hs_pad, hlens)
            hs_mask = hlens
        self.hs_pad = hs_pad

        # 1.5. transducer preparation related
        ys_in_pad, target, pred_len, target_len = prepare_loss_inputs(ys_pad, hs_mask)

        # 2. decoder
        if self.dtype == 'transformer':
            ys_mask = target_mask(ys_in_pad, self.blank_id)
            pred_pad, _ = self.decoder(ys_in_pad, ys_mask, hs_pad)
        else:
            if self.rnnt_mode == 'rnnt':
                pred_pad = self.decoder(hs_pad, ys_in_pad)
            else:
                pred_pad = self.decoder(hs_pad, ys_in_pad, pred_len)
        self.pred_pad = pred_pad

        # 3. loss computation
        loss = self.criterion(pred_pad, target, pred_len, target_len)

        self.loss = loss
        loss_data = float(self.loss)

        # 4. compute cer/wer
        if self.training or self.error_calculator is None:
            cer, wer = None, None
        else:
            cer, wer = self.error_calculator(hs_pad, ys_pad)

        if not math.isnan(loss_data):
            self.reporter.report(loss_data, cer, wer)
        else:
            logging.warning('loss (=%f) is not correct', loss_data)

        return self.loss

    def encode_transformer(self, x):
        """Encode acoustic features.

        Args:
            x (ndarray): input acoustic feature (T, D)

        Returns:
            x (torch.Tensor): encoded features (T, attention_dim)
        """
        self.eval()

        x = torch.as_tensor(x).unsqueeze(0)
        enc_output, _ = self.encoder(x, None)

        return enc_output.squeeze(0)

    def encode_rnn(self, x):
        """Encode acoustic features.

        Args:
            x (ndarray): input acoustic feature (T, D)

        Returns:
            x (torch.Tensor): encoded features (T, attention_dim)

        """
        self.eval()

        ilens = [x.shape[0]]

        x = x[::self.subsample[0], :]
        h = to_device(self, to_torch_tensor(x).float())
        hs = h.contiguous().unsqueeze(0)

        h, _, _ = self.encoder(hs, ilens)

        return h[0]

    def recognize(self, x, recog_args, char_list=None, rnnlm=None):
        """Recognize input features.

        Args:
            x (ndarray): input acoustic feature (T, D)
            recog_args (namespace): argument Namespace containing options
            char_list (list): list of characters
            rnnlm (torch.nn.Module): language model module

        Returns:
            y (list): n-best decoding results

        """
        if self.etype == 'transformer':
            h = self.encode_transformer(x)
        else:
            h = self.encode_rnn(x)
        params = [h, recog_args]

        if recog_args.beam_size == 1:
            nbest_hyps = self.decoder.recognize(*params)
        else:
            params.append(rnnlm)
            nbest_hyps = self.decoder.recognize_beam(*params)

        return nbest_hyps

    def calculate_all_attentions(self, xs_pad, ilens, ys_pad):
        """E2E attention calculation.

        Args:
            xs_pad (torch.Tensor): batch of padded input sequences (B, Tmax, idim)
            ilens (torch.Tensor): batch of lengths of input sequences (B)
            ys_pad (torch.Tensor): batch of padded character id sequence tensor (B, Lmax)

        Returns:
            ret (ndarray): attention weights with the following shape,
            1) multi-head case => attention weights (B, H, Lmax, Tmax),
            2) other case => attention weights (B, Lmax, Tmax).

        """
        if self.etype == 'transformer' and self.dtype != 'transformer' and \
           self.rnnt_mode == 'rnnt-att':
            raise NotImplementedError("Transformer encoder with rnn attention decoder"
                                      "is not supported yet.")
        elif self.etype != 'transformer' and self.dtype != 'transformer':
            if self.rnnt_mode == 'rnnt':
                return []
            else:
                with torch.no_grad():
                    hs_pad, hlens = xs_pad, ilens
                    hpad, hlens, _ = self.encoder(hs_pad, hlens)

                    ret = self.decoder.calculate_all_attentions(hpad, hlens, ys_pad)
        else:
            with torch.no_grad():
                self.forward(xs_pad, ilens, ys_pad)

                ret = dict()
                for name, m in self.named_modules():
                    if isinstance(m, MultiHeadedAttention):
                        ret[name] = m.attn.cpu().numpy()

        return ret

#!/usr/bin/env python3

"""Transducer related modules."""

import argparse
import logging
import math

import editdistance

import chainer
import numpy as np
import six
import torch

from chainer import reporter

from espnet.nets.asr_interface import ASRInterface

from espnet.nets.pytorch_backend.initialization import lecun_normal_init_parameters
from espnet.nets.pytorch_backend.initialization import set_forget_bias_to_one

from espnet.nets.pytorch_backend.nets_utils import to_device
from espnet.nets.pytorch_backend.nets_utils import to_torch_tensor

from espnet.nets.pytorch_backend.rnn.attentions import att_for
from espnet.nets.pytorch_backend.rnn.encoders import encoder_for

from espnet.nets.pytorch_backend.transducer.loss import TransLoss
from espnet.nets.pytorch_backend.transducer.rnn_decoders import decoder_for
from espnet.nets.pytorch_backend.transducer.utils import prepare_loss_inputs

from espnet.utils.cli_utils import strtobool


class Reporter(chainer.Chain):
    """A chainer reporter wrapper."""

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
        args (namespace): argument Namespace containing options

    """

    @staticmethod
    def add_arguments(parser):
        """Extend arguments for transducer."""
        group = parser.add_argument_group("transducer model setting")

        # encoder
        group.add_argument('--etype', default='blstmp', type=str,
                           choices=['lstm', 'blstm', 'lstmp', 'blstmp', 'vgglstmp', 'vggblstmp', 'vgglstm', 'vggblstm',
                                    'gru', 'bgru', 'grup', 'bgrup', 'vgggrup', 'vggbgrup', 'vgggru', 'vggbgru'],
                           help='Type of encoder network architecture')
        group.add_argument('--elayers', default=4, type=int,
                           help='Number of encoder layers (for shared recognition part in multi-speaker asr mode)')
        group.add_argument('--eunits', '-u', default=300, type=int,
                           help='Number of encoder hidden units')
        group.add_argument('--eprojs', default=320, type=int,
                           help='Number of encoder projection units')
        group.add_argument('--subsample', default="1", type=str,
                           help='Subsample input frames x_y_z means subsample every x frame at 1st layer, '
                           'every y frame at 2nd layer etc.')
        # attention
        group.add_argument('--atype', default='dot', type=str,
                           choices=['noatt', 'dot', 'add', 'location', 'coverage',
                                    'coverage_location', 'location2d', 'location_recurrent',
                                    'multi_head_dot', 'multi_head_add', 'multi_head_loc',
                                    'multi_head_multi_res_loc'],
                           help='Type of attention architecture')
        group.add_argument('--adim', default=320, type=int,
                           help='Number of attention transformation dimensions')
        group.add_argument('--awin', default=5, type=int,
                           help='Window size for location2d attention')
        group.add_argument('--aheads', default=4, type=int,
                           help='Number of heads for multi head attention')
        group.add_argument('--aconv-chans', default=-1, type=int,
                           help='Number of attention convolution channels \
                           (negative value indicates no location-aware attention)')
        group.add_argument('--aconv-filts', default=100, type=int,
                           help='Number of attention convolution filters \
                           (negative value indicates no location-aware attention)')
        group.add_argument('--dropout-rate', default=0.0, type=float,
                           help='Dropout rate for the encoder')
        # decoder
        group.add_argument('--dtype', default='lstm', type=str,
                           choices=['lstm', 'gru'],
                           help='Type of decoder network architecture')
        group.add_argument('--dlayers', default=1, type=int,
                           help='Number of decoder layers')
        group.add_argument('--dunits', default=320, type=int,
                           help='Number of decoder hidden units')
        group.add_argument('--dropout-rate-decoder', default=0.0, type=float,
                           help='Dropout rate for the decoder')
        # prediction
        group.add_argument('--dec-embed-dim', default=320, type=int,
                           help='Number of decoder embeddings dimensions')
        group.add_argument('--dropout-rate-embed-decoder', default=0.0, type=float,
                           help='Dropout rate for the decoder embeddings')
        # transducer
        group.add_argument('--rnnt-type', default='warp-transducer', type=str,
                           choices=['warp-transducer'],
                           help='Type of transducer implementation to calculate loss.')
        group.add_argument('--rnnt-mode', default='rnnt', type=str, choices=['rnnt', 'rnnt-att'],
                           help='RNN-Transducing mode')
        group.add_argument('--joint-dim', default=320, type=int,
                           help='Number of dimensions in joint space')
        group.add_argument('--score-norm-transducer', type=strtobool, nargs='?',
                           default=True,
                           help='Normalize transducer scores by length')

        return parser

    def __init__(self, idim, odim, args):
        """Initialize transducer modules.

        Args:
            idim (int): dimension of inputs
            odim (int): dimension of outputs
            args (Namespace): argument Namespace containing options

        """
        super(E2E, self).__init__()
        torch.nn.Module.__init__(self)
        self.rnnt_mode = args.rnnt_mode
        self.etype = args.etype
        self.verbose = args.verbose
        self.char_list = args.char_list
        self.outdir = args.outdir
        self.space = args.sym_space
        self.blank = args.sym_blank
        self.reporter = Reporter()
        self.beam_size = args.beam_size

        # note that eos is the same as sos (equivalent ID)
        self.sos = odim - 1
        self.eos = odim - 1
        self.blank_id = 0
        self.ignore_id = -1

        subsample = np.ones(args.elayers + 1, dtype=np.int)
        if args.etype.endswith("p") and not args.etype.startswith("vgg"):
            ss = args.subsample.split("_")
            for j in range(min(args.elayers + 1, len(ss))):
                subsample[j] = int(ss[j])
        else:
            logging.warning(
                'Subsampling is not performed for vgg*. It is performed in max pooling layers at CNN.')
            logging.info('subsample: ' + ' '.join([str(x) for x in subsample]))
        self.subsample = subsample

        self.enc = encoder_for(args, idim, subsample)

        if args.rnnt_mode == 'rnnt-att':
            self.att = att_for(args)
            self.dec = decoder_for(args, odim, self.att)
        else:
            self.dec = decoder_for(args, odim)

        self.init_like_chainer()

        self.criterion = TransLoss(args.rnnt_type, self.blank_id)

        # options for beam search
        if 'report_cer' in vars(args) and (args.report_cer or args.report_wer):
            recog_args = {'beam_size': args.beam_size, 'nbest': args.nbest, 'space': args.sym_space,
                          'score_norm_transducer': args.score_norm_transducer}

            self.recog_args = argparse.Namespace(**recog_args)
            self.report_cer = args.report_cer
            self.report_wer = args.report_wer
        else:
            self.report_cer = False
            self.report_wer = False

        self.logzero = -10000000000.0
        self.rnnlm = None
        self.loss = None

    def init_like_chainer(self):
        """Initialize weight like chainer.

        chainer basically uses LeCun way: W ~ Normal(0, fan_in ** -0.5), b = 0
        pytorch basically uses W, b ~ Uniform(-fan_in**-0.5, fan_in**-0.5)

        however, there are two exceptions as far as I know.
        - EmbedID.W ~ Normal(0, 1)
        - LSTM.upward.b[forget_gate_range] = 1 (but not used in NStepLSTM)

        """
        lecun_normal_init_parameters(self)

        if self.rnnt_mode == 'rnnt-att':
            # embed weight ~ Normal(0, 1)
            self.dec.embed.weight.data.normal_(0, 1)
            # forget-bias = 1.0
            # https://discuss.pytorch.org/t/set-forget-gate-bias-of-lstm/1745
            for l in range(len(self.dec.decoder)):
                set_forget_bias_to_one(self.dec.decoder[l].bias_ih)
        else:
            self.dec.embed.weight.data.normal_(0, 1)

    def forward(self, xs_pad, ilens, ys_pad):
        """E2E forward.

        Args:
            xs_pad (torch.Tensor): batch of padded input sequences (B, Tmax, idim)
            ilens (torch.Tensor): batch of lengths of input sequences (B)
            ys_pad (torch.Tensor): batch of padded character id sequence tensor (B, Lmax)

        Returns:
               loss (torch.Tensor): transducer loss value

        """
        # 1. encoder
        hs_pad, hlens = xs_pad, ilens
        hs_pad, hlens, _ = self.enc(hs_pad, hlens)

        # 1.5 transducer preparation related
        ys_in_pad, target, pred_len, target_len = prepare_loss_inputs(ys_pad, hlens)

        # 2. decoder
        if self.rnnt_mode == 'rnnt':
            z = self.dec(hs_pad, ys_in_pad)
        else:
            z = self.dec(hs_pad, ys_in_pad, hlens)

        # 3. loss computation
        loss = self.criterion(z, target, pred_len, target_len)

        # 4. compute cer/wer
        # note: not recommended outside debugging right now,
        # the training time is hugely impacted.
        if self.training or not (self.report_cer or self.report_wer):
            cer, wer = 0.0, 0.0
        else:
            word_eds, word_ref_lens, char_eds, char_ref_lens = [], [], [], []

            batchsize = int(hs_pad.size(0))
            batch_nbest = []

            for b in six.moves.range(batchsize):
                if self.beam_size == 1:
                    nbest_hyps = self.dec.recognize(hs_pad[b], self.recog_args)
                else:
                    nbest_hyps = self.dec.recognize_beam(hs_pad[b], self.recog_args)

                batch_nbest.append(nbest_hyps)

            y_hats = [nbest_hyp[0]['yseq'][1:] for nbest_hyp in batch_nbest]

            for i, y_hat in enumerate(y_hats):
                y_true = ys_pad[i]

                seq_hat = [self.char_list[int(idx)] for idx in y_hat]
                seq_true = [self.char_list[int(idx)] for idx in y_true if int(idx) != -1]
                seq_hat_text = "".join(seq_hat).replace(self.recog_args.space, ' ')
                seq_true_text = "".join(seq_true).replace(self.recog_args.space, ' ')

                hyp_words = seq_hat_text.split()
                ref_words = seq_true_text.split()
                word_eds.append(editdistance.eval(hyp_words, ref_words))
                word_ref_lens.append(len(ref_words))

                hyp_chars = seq_hat_text.replace(' ', '')
                ref_chars = seq_true_text.replace(' ', '')
                char_eds.append(editdistance.eval(hyp_chars, ref_chars))
                char_ref_lens.append(len(ref_chars))
            wer = 0.0 if not self.report_wer else float(sum(word_eds)) / sum(word_ref_lens)
            cer = 0.0 if not self.report_cer else float(sum(char_eds)) / sum(char_ref_lens)

        self.loss = loss
        loss_data = float(self.loss)

        if not math.isnan(loss_data):
            self.reporter.report(loss_data, cer, wer)
        else:
            logging.warning('loss (=%f) is not correct', loss_data)

        return self.loss

    def recognize(self, x, recog_args, char_list, rnnlm=None):
        """E2E recognize.

        Args:
            x (ndarray): input acoustic feature (T, D)
            recog_args (namespace): argument Namespace containing options
            char_list (list): list of characters
            rnnlm (torch.nn.Module): language model module

        Returns:
           y (list): n-best decoding results

        """
        prev = self.training
        self.eval()
        ilens = [x.shape[0]]

        # subsample frame
        x = x[::self.subsample[0], :]
        h = to_device(self, to_torch_tensor(x).float())
        # make a utt list (1) to use the same interface for encoder
        hs = h.contiguous().unsqueeze(0)

        hs, hlens = hs, ilens

        # 1. Encoder
        h, _, _ = self.enc(hs, hlens)

        # 2. Decoder
        if recog_args.beam_size == 1:
            y = self.dec.recognize(h[0], recog_args)
        else:
            y = self.dec.recognize_beam(h[0], recog_args, rnnlm)

        if prev:
            self.train()

        return y

    def calculate_all_attentions(self, xs_pad, ilens, ys_pad):
        """E2E attention calculation.

        Args:
            xs_pad (torch.Tensor): batch of padded input sequences (B, Tmax, idim)
            ilens (torch.Tensor): batch of lengths of input sequences (B)
            ys_pad (torch.Tensor): batch of padded character id sequence tensor (B, Lmax)

        Returns:
            att_ws (ndarray): attention weights with the following shape,
                1) multi-head case => attention weights (B, H, Lmax, Tmax),
                2) other case => attention weights (B, Lmax, Tmax).

        """
        if self.rnnt_mode == 'rnnt':
            return []

        with torch.no_grad():
            hs_pad, hlens = xs_pad, ilens

            # encoder
            hpad, hlens, _ = self.enc(hs_pad, hlens)

            # decoder
            att_ws = self.dec.calculate_all_attentions(hpad, hlens, ys_pad)

        return att_ws

    def subsample_frames(self, x):
        """Subsample frames."""
        x = x[::self.subsample[0], :]
        ilen = [x.shape[0]]
        h = to_device(self, torch.from_numpy(
            np.array(x, dtype=np.float32)))
        h.contiguous()
        return h, ilen

#!/usr/bin/env python3

"""Define e2e module for multi-encoder network. https://arxiv.org/pdf/1811.04903.pdf."""
# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
# Copyright 2017 Johns Hopkins University (Ruizhi Li)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


import argparse
import logging
import math
import os

import editdistance

import chainer
import numpy as np
import torch

from itertools import groupby

from chainer import reporter

from espnet.nets.asr_interface import ASRInterface
from espnet.nets.e2e_asr_common import label_smoothing_dist
from espnet.nets.pytorch_backend.ctc import ctc_for
from espnet.nets.pytorch_backend.nets_utils import get_subsample
from espnet.nets.pytorch_backend.nets_utils import pad_list
from espnet.nets.pytorch_backend.nets_utils import to_device
from espnet.nets.pytorch_backend.nets_utils import to_torch_tensor
from espnet.nets.pytorch_backend.rnn.attentions import att_for
from espnet.nets.pytorch_backend.rnn.decoders import decoder_for
from espnet.nets.pytorch_backend.rnn.encoders import encoder_for
from espnet.nets.scorers.ctc import CTCPrefixScorer
from espnet.utils.cli_utils import strtobool

CTC_LOSS_THRESHOLD = 10000


class Reporter(chainer.Chain):
    """Define a chainer reporter wrapper."""

    def report(self, loss_ctc_list, loss_att, acc, cer_ctc_list, cer, wer, mtl_loss):
        """Define a chainer reporter function."""
        # loss_ctc_list = [weighted CTC, CTC1, CTC2, ... CTCN]
        # cer_ctc_list = [weighted cer_ctc, cer_ctc_1, cer_ctc_2, ... cer_ctc_N]
        num_encs = len(loss_ctc_list) - 1
        reporter.report({'loss_ctc': loss_ctc_list[0]}, self)
        for i in range(num_encs):
            reporter.report({'loss_ctc{}'.format(i + 1): loss_ctc_list[i + 1]}, self)
        reporter.report({'loss_att': loss_att}, self)
        reporter.report({'acc': acc}, self)
        reporter.report({'cer_ctc': cer_ctc_list[0]}, self)
        for i in range(num_encs):
            reporter.report({'cer_ctc{}'.format(i + 1): cer_ctc_list[i + 1]}, self)
        reporter.report({'cer': cer}, self)
        reporter.report({'wer': wer}, self)
        logging.info('mtl loss:' + str(mtl_loss))
        reporter.report({'loss': mtl_loss}, self)


class E2E(ASRInterface, torch.nn.Module):
    """E2E module.

    :param List idims: List of dimensions of inputs
    :param int odim: dimension of outputs
    :param Namespace args: argument Namespace containing options

    """

    @staticmethod
    def add_arguments(parser):
        """Add arguments for multi-encoder setting."""
        E2E.encoder_add_arguments(parser)
        E2E.attention_add_arguments(parser)
        E2E.decoder_add_arguments(parser)
        E2E.ctc_add_arguments(parser)
        return parser

    @staticmethod
    def encoder_add_arguments(parser):
        """Add arguments for encoders in multi-encoder setting."""
        group = parser.add_argument_group("E2E encoder setting")
        group.add_argument('--etype', action='append', type=str,
                           choices=['lstm', 'blstm', 'lstmp', 'blstmp', 'vgglstmp', 'vggblstmp', 'vgglstm', 'vggblstm',
                                    'gru', 'bgru', 'grup', 'bgrup', 'vgggrup', 'vggbgrup', 'vgggru', 'vggbgru'],
                           help='Type of encoder network architecture')
        group.add_argument('--elayers', type=int, action='append',
                           help='Number of encoder layers (for shared recognition part in multi-speaker asr mode)')
        group.add_argument('--eunits', '-u', type=int, action='append',
                           help='Number of encoder hidden units')
        group.add_argument('--eprojs', default=320, type=int,
                           help='Number of encoder projection units')
        group.add_argument('--subsample', type=str, action='append',
                           help='Subsample input frames x_y_z means subsample every x frame at 1st layer, '
                                'every y frame at 2nd layer etc.')
        return parser

    @staticmethod
    def attention_add_arguments(parser):
        """Add arguments for attentions in multi-encoder setting."""
        group = parser.add_argument_group("E2E attention setting")
        # attention
        group.add_argument('--atype', type=str, action='append',
                           choices=['noatt', 'dot', 'add', 'location', 'coverage',
                                    'coverage_location', 'location2d', 'location_recurrent',
                                    'multi_head_dot', 'multi_head_add', 'multi_head_loc',
                                    'multi_head_multi_res_loc'],
                           help='Type of attention architecture')
        group.add_argument('--adim', type=int, action='append',
                           help='Number of attention transformation dimensions')
        group.add_argument('--awin', type=int, action='append',
                           help='Window size for location2d attention')
        group.add_argument('--aheads', type=int, action='append',
                           help='Number of heads for multi head attention')
        group.add_argument('--aconv-chans', type=int, action='append',
                           help='Number of attention convolution channels \
                           (negative value indicates no location-aware attention)')
        group.add_argument('--aconv-filts', type=int, action='append',
                           help='Number of attention convolution filters \
                           (negative value indicates no location-aware attention)')
        group.add_argument('--dropout-rate', type=float, action='append',
                           help='Dropout rate for the encoder')
        # hierarchical attention network (HAN)
        group.add_argument('--han-type', default='dot', type=str,
                           choices=['noatt', 'dot', 'add', 'location', 'coverage',
                                    'coverage_location', 'location2d', 'location_recurrent',
                                    'multi_head_dot', 'multi_head_add', 'multi_head_loc',
                                    'multi_head_multi_res_loc'],
                           help='Type of attention architecture (multi-encoder asr mode only)')
        group.add_argument('--han-dim', default=320, type=int,
                           help='Number of attention transformation dimensions in HAN')
        group.add_argument('--han-win', default=5, type=int,
                           help='Window size for location2d attention in HAN')
        group.add_argument('--han-heads', default=4, type=int,
                           help='Number of heads for multi head attention in HAN')
        group.add_argument('--han-conv-chans', default=-1, type=int,
                           help='Number of attention convolution channels  in HAN \
                           (negative value indicates no location-aware attention)')
        group.add_argument('--han-conv-filts', default=100, type=int,
                           help='Number of attention convolution filters in HAN \
                           (negative value indicates no location-aware attention)')
        return parser

    @staticmethod
    def decoder_add_arguments(parser):
        """Add arguments for decoder in multi-encoder setting."""
        group = parser.add_argument_group("E2E decoder setting")
        group.add_argument('--dtype', default='lstm', type=str,
                           choices=['lstm', 'gru'],
                           help='Type of decoder network architecture')
        group.add_argument('--dlayers', default=1, type=int,
                           help='Number of decoder layers')
        group.add_argument('--dunits', default=320, type=int,
                           help='Number of decoder hidden units')
        group.add_argument('--dropout-rate-decoder', default=0.0, type=float,
                           help='Dropout rate for the decoder')
        group.add_argument('--sampling-probability', default=0.0, type=float,
                           help='Ratio of predicted labels fed back to decoder')
        group.add_argument('--lsm-type', const='', default='', type=str, nargs='?',
                           choices=['', 'unigram'],
                           help='Apply label smoothing with a specified distribution type')
        return parser

    @staticmethod
    def ctc_add_arguments(parser):
        """Add arguments for ctc in multi-encoder setting."""
        group = parser.add_argument_group("E2E multi-ctc setting")
        group.add_argument('--share-ctc', type=strtobool, default=False,
                           help='The flag to switch to share ctc across multiple encoders '
                                '(multi-encoder asr mode only).')
        group.add_argument('--weights-ctc-train', type=float, action='append',
                           help='ctc weight assigned to each encoder during training.')
        group.add_argument('--weights-ctc-dec', type=float, action='append',
                           help='ctc weight assigned to each encoder during decoding.')
        return parser

    def __init__(self, idims, odim, args):
        """Initialize this class with python-level args.

        Args:
            idims (list): list of the number of an input feature dim.
            odim (int): The number of output vocab.
            args (Namespace): arguments

        """
        super(E2E, self).__init__()
        torch.nn.Module.__init__(self)
        self.mtlalpha = args.mtlalpha
        assert 0.0 <= self.mtlalpha <= 1.0, "mtlalpha should be [0.0, 1.0]"
        self.verbose = args.verbose
        # NOTE: for self.build method
        args.char_list = getattr(args, "char_list", None)
        self.char_list = args.char_list
        self.outdir = args.outdir
        self.space = args.sym_space
        self.blank = args.sym_blank
        self.reporter = Reporter()
        self.num_encs = args.num_encs
        self.share_ctc = args.share_ctc

        # below means the last number becomes eos/sos ID
        # note that sos/eos IDs are identical
        self.sos = odim - 1
        self.eos = odim - 1

        # subsample info
        self.subsample_list = get_subsample(args, mode='asr', arch='rnn_mulenc')

        # label smoothing info
        if args.lsm_type and os.path.isfile(args.train_json):
            logging.info("Use label smoothing with " + args.lsm_type)
            labeldist = label_smoothing_dist(odim, args.lsm_type, transcript=args.train_json)
        else:
            labeldist = None

        # speech translation related
        self.replace_sos = getattr(args, "replace_sos", False)  # use getattr to keep compatibility

        self.frontend = None

        # encoder
        self.enc = encoder_for(args, idims, self.subsample_list)
        # ctc
        self.ctc = ctc_for(args, odim)
        # attention
        self.att = att_for(args)
        # hierarchical attention network
        han = att_for(args, han_mode=True)
        self.att.append(han)
        # decoder
        self.dec = decoder_for(args, odim, self.sos, self.eos, self.att, labeldist)

        if args.mtlalpha > 0 and self.num_encs > 1:
            # weights-ctc, e.g. ctc_loss = w_1*ctc_1_loss + w_2 * ctc_2_loss + w_N * ctc_N_loss
            self.weights_ctc_train = args.weights_ctc_train / np.sum(args.weights_ctc_train)  # normalize
            self.weights_ctc_dec = args.weights_ctc_dec / np.sum(args.weights_ctc_dec)  # normalize
            logging.info(
                'ctc weights (training during training): ' + ' '.join([str(x) for x in self.weights_ctc_train]))
            logging.info('ctc weights (decoding during training): ' + ' '.join([str(x) for x in self.weights_ctc_dec]))
        else:
            self.weights_ctc_dec = [1.0]
            self.weights_ctc_train = [1.0]

        # weight initialization
        self.init_like_chainer()

        # options for beam search
        if args.report_cer or args.report_wer:
            recog_args = {'beam_size': args.beam_size, 'penalty': args.penalty,
                          'ctc_weight': args.ctc_weight, 'maxlenratio': args.maxlenratio,
                          'minlenratio': args.minlenratio, 'lm_weight': args.lm_weight,
                          'rnnlm': args.rnnlm, 'nbest': args.nbest,
                          'space': args.sym_space, 'blank': args.sym_blank,
                          'tgt_lang': False, 'ctc_weights_dec': self.weights_ctc_dec}

            self.recog_args = argparse.Namespace(**recog_args)
            self.report_cer = args.report_cer
            self.report_wer = args.report_wer
        else:
            self.report_cer = False
            self.report_wer = False
        self.rnnlm = None

        self.logzero = -10000000000.0
        self.loss = None
        self.acc = None

    def init_like_chainer(self):
        """Initialize weight like chainer.

        chainer basically uses LeCun way: W ~ Normal(0, fan_in ** -0.5), b = 0
        pytorch basically uses W, b ~ Uniform(-fan_in**-0.5, fan_in**-0.5)

        however, there are two exceptions as far as I know.
        - EmbedID.W ~ Normal(0, 1)
        - LSTM.upward.b[forget_gate_range] = 1 (but not used in NStepLSTM)
        """
        def lecun_normal_init_parameters(module):
            for p in module.parameters():
                data = p.data
                if data.dim() == 1:
                    # bias
                    data.zero_()
                elif data.dim() == 2:
                    # linear weight
                    n = data.size(1)
                    stdv = 1. / math.sqrt(n)
                    data.normal_(0, stdv)
                elif data.dim() in (3, 4):
                    # conv weight
                    n = data.size(1)
                    for k in data.size()[2:]:
                        n *= k
                    stdv = 1. / math.sqrt(n)
                    data.normal_(0, stdv)
                else:
                    raise NotImplementedError

        def set_forget_bias_to_one(bias):
            n = bias.size(0)
            start, end = n // 4, n // 2
            bias.data[start:end].fill_(1.)

        lecun_normal_init_parameters(self)
        # exceptions
        # embed weight ~ Normal(0, 1)
        self.dec.embed.weight.data.normal_(0, 1)
        # forget-bias = 1.0
        # https://discuss.pytorch.org/t/set-forget-gate-bias-of-lstm/1745
        for l in range(len(self.dec.decoder)):
            set_forget_bias_to_one(self.dec.decoder[l].bias_ih)

    def forward(self, xs_pad_list, ilens_list, ys_pad):
        """E2E forward.

        :param List xs_pad_list: list of batch (torch.Tensor) of padded input sequences
                                [(B, Tmax_1, idim), (B, Tmax_2, idim),..]
        :param List ilens_list: list of batch (torch.Tensor) of lengths of input sequences [(B), (B), ..]
        :param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, Lmax)
        :return: loss value
        :rtype: torch.Tensor
        """
        if self.replace_sos:
            tgt_lang_ids = ys_pad[:, 0:1]
            ys_pad = ys_pad[:, 1:]  # remove target language ID in the beginning
        else:
            tgt_lang_ids = None

        hs_pad_list, hlens_list, self.loss_ctc_list = [], [], []
        for idx in range(self.num_encs):
            # 1. Encoder
            hs_pad, hlens, _ = self.enc[idx](xs_pad_list[idx], ilens_list[idx])

            # 2. CTC loss
            if self.mtlalpha == 0:
                self.loss_ctc_list.append(None)
            else:
                ctc_idx = 0 if self.share_ctc else idx
                loss_ctc = self.ctc[ctc_idx](hs_pad, hlens, ys_pad)
                self.loss_ctc_list.append(loss_ctc)
            hs_pad_list.append(hs_pad)
            hlens_list.append(hlens)

        # 3. attention loss
        if self.mtlalpha == 1:
            self.loss_att, acc = None, None
        else:
            self.loss_att, acc, _ = self.dec(hs_pad_list, hlens_list, ys_pad, lang_ids=tgt_lang_ids)
        self.acc = acc

        # 4. compute cer without beam search
        if self.mtlalpha == 0 or self.char_list is None:
            cer_ctc_list = [None] * (self.num_encs + 1)
        else:
            cer_ctc_list = []
            for ind in range(self.num_encs):
                cers = []
                ctc_idx = 0 if self.share_ctc else ind
                y_hats = self.ctc[ctc_idx].argmax(hs_pad_list[ind]).data
                for i, y in enumerate(y_hats):
                    y_hat = [x[0] for x in groupby(y)]
                    y_true = ys_pad[i]

                    seq_hat = [self.char_list[int(idx)] for idx in y_hat if int(idx) != -1]
                    seq_true = [self.char_list[int(idx)] for idx in y_true if int(idx) != -1]
                    seq_hat_text = "".join(seq_hat).replace(self.space, ' ')
                    seq_hat_text = seq_hat_text.replace(self.blank, '')
                    seq_true_text = "".join(seq_true).replace(self.space, ' ')

                    hyp_chars = seq_hat_text.replace(' ', '')
                    ref_chars = seq_true_text.replace(' ', '')
                    if len(ref_chars) > 0:
                        cers.append(editdistance.eval(hyp_chars, ref_chars) / len(ref_chars))

                cer_ctc = sum(cers) / len(cers) if cers else None
                cer_ctc_list.append(cer_ctc)
            cer_ctc_weighted = np.sum([item * self.weights_ctc_train[i] for i, item in enumerate(cer_ctc_list)])
            cer_ctc_list = [float(cer_ctc_weighted)] + [float(item) for item in cer_ctc_list]

        # 5. compute cer/wer
        if self.training or not (self.report_cer or self.report_wer):
            cer, wer = 0.0, 0.0
            # oracle_cer, oracle_wer = 0.0, 0.0
        else:
            if self.recog_args.ctc_weight > 0.0:
                lpz_list = []
                for idx in range(self.num_encs):
                    ctc_idx = 0 if self.share_ctc else idx
                    lpz = self.ctc[ctc_idx].log_softmax(hs_pad_list[idx]).data
                    lpz_list.append(lpz)
            else:
                lpz_list = None

            word_eds, word_ref_lens, char_eds, char_ref_lens = [], [], [], []
            nbest_hyps = self.dec.recognize_beam_batch(
                hs_pad_list, hlens_list, lpz_list,
                self.recog_args, self.char_list,
                self.rnnlm,
                lang_ids=tgt_lang_ids.squeeze(1).tolist() if self.replace_sos else None)
            # remove <sos> and <eos>
            y_hats = [nbest_hyp[0]['yseq'][1:-1] for nbest_hyp in nbest_hyps]
            for i, y_hat in enumerate(y_hats):
                y_true = ys_pad[i]

                seq_hat = [self.char_list[int(idx)] for idx in y_hat if int(idx) != -1]
                seq_true = [self.char_list[int(idx)] for idx in y_true if int(idx) != -1]
                seq_hat_text = "".join(seq_hat).replace(self.recog_args.space, ' ')
                seq_hat_text = seq_hat_text.replace(self.recog_args.blank, '')
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

        alpha = self.mtlalpha
        if alpha == 0:
            self.loss = self.loss_att
            loss_att_data = float(self.loss_att)
            loss_ctc_data_list = [None] * (self.num_encs + 1)
        elif alpha == 1:
            self.loss = torch.sum(torch.cat(
                [(item * self.weights_ctc_train[i]).unsqueeze(0) for i, item in enumerate(self.loss_ctc_list)]))
            loss_att_data = None
            loss_ctc_data_list = [float(self.loss)] + [float(item) for item in self.loss_ctc_list]
        else:
            self.loss_ctc = torch.sum(torch.cat(
                [(item * self.weights_ctc_train[i]).unsqueeze(0) for i, item in enumerate(self.loss_ctc_list)]))
            self.loss = alpha * self.loss_ctc + (1 - alpha) * self.loss_att
            loss_att_data = float(self.loss_att)
            loss_ctc_data_list = [float(self.loss_ctc)] + [float(item) for item in self.loss_ctc_list]

        loss_data = float(self.loss)
        if loss_data < CTC_LOSS_THRESHOLD and not math.isnan(loss_data):
            self.reporter.report(loss_ctc_data_list, loss_att_data, acc, cer_ctc_list, cer, wer, loss_data)
        else:
            logging.warning('loss (=%f) is not correct', loss_data)
        return self.loss

    def scorers(self):
        """Get scorers for `beam_search` (optional).

        Returns:
            dict[str, ScorerInterface]: dict of `ScorerInterface` objects

        """
        return dict(decoder=self.dec, ctc=CTCPrefixScorer(self.ctc, self.eos))

    def encode(self, x_list):
        """Encode feature.

        Args:
            x_list (list): input feature [(T1, D), (T2, D), ... ]
        Returns:
            list
                encoded feature [(T1, D), (T2, D), ... ]

        """
        self.eval()
        ilens_list = [[x_list[idx].shape[0]] for idx in range(self.num_encs)]

        # subsample frame
        x_list = [x_list[idx][::self.subsample_list[idx][0], :] for idx in range(self.num_encs)]
        p = next(self.parameters())
        x_list = [torch.as_tensor(x_list[idx], device=p.device, dtype=p.dtype) for idx in range(self.num_encs)]
        # make a utt list (1) to use the same interface for encoder
        xs_list = [x_list[idx].contiguous().unsqueeze(0) for idx in range(self.num_encs)]

        # 1. encoder
        hs_list = []
        for idx in range(self.num_encs):
            hs, _, _ = self.enc[idx](xs_list[idx], ilens_list[idx])
            hs_list.append(hs[0])
        return hs_list

    def recognize(self, x_list, recog_args, char_list, rnnlm=None):
        """E2E beam search.

        :param list of ndarray x: list of input acoustic feature [(T1, D), (T2,D),...]
        :param Namespace recog_args: argument Namespace containing options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """
        hs_list = self.encode(x_list)
        # calculate log P(z_t|X) for CTC scores
        if recog_args.ctc_weight > 0.0:
            if self.share_ctc:
                lpz_list = [self.ctc[0].log_softmax(hs_list[idx].unsqueeze(0))[0] for idx in range(self.num_encs)]
            else:
                lpz_list = [self.ctc[idx].log_softmax(hs_list[idx].unsqueeze(0))[0] for idx in range(self.num_encs)]
        else:
            lpz_list = None

        # 2. Decoder
        # decode the first utterance
        y = self.dec.recognize_beam(hs_list, lpz_list, recog_args, char_list, rnnlm)
        return y

    def recognize_batch(self, xs_list, recog_args, char_list, rnnlm=None):
        """E2E beam search.

        :param list xs_list: list of list of input acoustic feature arrays
                [[(T1_1, D), (T1_2, D), ...],[(T2_1, D), (T2_2, D), ...], ...]
        :param Namespace recog_args: argument Namespace containing options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """
        prev = self.training
        self.eval()
        ilens_list = [np.fromiter((xx.shape[0] for xx in xs_list[idx]), dtype=np.int64) for idx in range(self.num_encs)]

        # subsample frame
        xs_list = [[xx[::self.subsample_list[idx][0], :] for xx in xs_list[idx]] for idx in range(self.num_encs)]

        xs_list = [[to_device(self, to_torch_tensor(xx).float()) for xx in xs_list[idx]] for idx in
                   range(self.num_encs)]
        xs_pad_list = [pad_list(xs_list[idx], 0.0) for idx in range(self.num_encs)]

        # 1. Encoder
        hs_pad_list, hlens_list = [], []
        for idx in range(self.num_encs):
            hs_pad, hlens, _ = self.enc[idx](xs_pad_list[idx], ilens_list[idx])
            hs_pad_list.append(hs_pad)
            hlens_list.append(hlens)

        # calculate log P(z_t|X) for CTC scores
        if recog_args.ctc_weight > 0.0:
            if self.share_ctc:
                lpz_list = [self.ctc[0].log_softmax(hs_pad_list[idx]) for idx in range(self.num_encs)]
            else:
                lpz_list = [self.ctc[idx].log_softmax(hs_pad_list[idx]) for idx in range(self.num_encs)]
            normalize_score = False
        else:
            lpz_list = None
            normalize_score = True

        # 2. Decoder
        hlens_list = [torch.tensor(list(map(int, hlens_list[idx]))) for idx in
                      range(self.num_encs)]  # make sure hlens is tensor
        y = self.dec.recognize_beam_batch(hs_pad_list, hlens_list, lpz_list, recog_args, char_list,
                                          rnnlm, normalize_score=normalize_score)

        if prev:
            self.train()
        return y

    def calculate_all_attentions(self, xs_pad_list, ilens_list, ys_pad):
        """E2E attention calculation.

        :param List xs_pad_list: list of batch (torch.Tensor) of padded input sequences
                                [(B, Tmax_1, idim), (B, Tmax_2, idim),..]
        :param List ilens_list: list of batch (torch.Tensor) of lengths of input sequences [(B), (B), ..]
        :param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, Lmax)
        :return: attention weights with the following shape,
            1) multi-head case => attention weights (B, H, Lmax, Tmax),
            2) multi-encoder case => [(B, Lmax, Tmax1), (B, Lmax, Tmax2), ..., (B, Lmax, NumEncs)]
            3) other case => attention weights (B, Lmax, Tmax).
        :rtype: float ndarray or list
        """
        with torch.no_grad():
            # 1. Encoder
            if self.replace_sos:
                tgt_lang_ids = ys_pad[:, 0:1]
                ys_pad = ys_pad[:, 1:]  # remove target language ID in the beggining
            else:
                tgt_lang_ids = None

            hs_pad_list, hlens_list = [], []
            for idx in range(self.num_encs):
                hs_pad, hlens, _ = self.enc[idx](xs_pad_list[idx], ilens_list[idx])
                hs_pad_list.append(hs_pad)
                hlens_list.append(hlens)

            # 2. Decoder
            att_ws = self.dec.calculate_all_attentions(hs_pad_list, hlens_list, ys_pad, lang_ids=tgt_lang_ids)

        return att_ws

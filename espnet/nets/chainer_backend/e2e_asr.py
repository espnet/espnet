#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


import logging
import math

import chainer
from chainer import reporter
import numpy as np

from espnet.nets.asr_interface import ASRInterface
from espnet.nets.chainer_backend.attentions import att_for
from espnet.nets.chainer_backend.ctc import ctc_for
from espnet.nets.chainer_backend.decoders import decoder_for
from espnet.nets.chainer_backend.encoders import encoder_for
from espnet.nets.e2e_asr_common import label_smoothing_dist

CTC_LOSS_THRESHOLD = 10000


class E2E(ASRInterface, chainer.Chain):
    @staticmethod
    def add_arguments(parser):
        E2E.encoder_add_arguments(parser)
        E2E.attention_add_arguments(parser)
        E2E.decoder_add_arguments(parser)
        E2E.loss_add_arguments(parser)
        E2E.recognition_add_arguments(parser)
        return parser

    @staticmethod
    def encoder_add_arguments(parser):
        group = parser.add_argument_group("E2E encoder setting")
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
        return parser

    @staticmethod
    def attention_add_arguments(parser):
        group = parser.add_argument_group("E2E attention setting")
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
        return parser

    @staticmethod
    def decoder_add_arguments(parser):
        group = parser.add_argument_group("E2E encoder setting")
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
        return parser

    @staticmethod
    def loss_add_arguments(parser):
        group = parser.add_argument_group("E2E loss setting")
        group.add_argument('--ctc_type', default='warpctc', type=str,
                           choices=['builtin', 'warpctc'],
                           help='Type of CTC implementation to calculate loss.')
        group.add_argument('--mtlalpha', default=0.5, type=float,
                           help='Multitask learning coefficient, alpha: alpha*ctc_loss + (1-alpha)*att_loss ')
        group.add_argument('--lsm-type', const='', default='', type=str, nargs='?', choices=['', 'unigram'],
                           help='Apply label smoothing with a specified distribution type')
        group.add_argument('--lsm-weight', default=0.0, type=float,
                           help='Label smoothing weight')
        return parser

    @staticmethod
    def recognition_add_arguments(parser):
        group = parser.add_argument_group("E2E recognition setting")
        # recognition options to compute CER/WER
        group.add_argument('--report-cer', default=False, action='store_true',
                           help='Compute CER on development set')
        group.add_argument('--report-wer', default=False, action='store_true',
                           help='Compute WER on development set')
        group.add_argument('--nbest', type=int, default=1,
                           help='Output N-best hypotheses')
        group.add_argument('--beam-size', type=int, default=4,
                           help='Beam size')
        group.add_argument('--penalty', default=0.0, type=float,
                           help='Incertion penalty')
        group.add_argument('--maxlenratio', default=0.0, type=float,
                           help="""Input length ratio to obtain max output length.
                           If maxlenratio=0.0 (default), it uses a end-detect function
                           to automatically find maximum hypothesis lengths""")
        group.add_argument('--minlenratio', default=0.0, type=float,
                           help='Input length ratio to obtain min output length')
        group.add_argument('--ctc-weight', default=0.3, type=float,
                           help='CTC weight in joint decoding')
        group.add_argument('--rnnlm', type=str, default=None,
                           help='RNNLM model file to read')
        group.add_argument('--rnnlm-conf', type=str, default=None,
                           help='RNNLM model config file to read')
        group.add_argument('--lm-weight', default=0.1, type=float,
                           help='RNNLM weight.')
        group.add_argument('--sym-space', default='<space>', type=str,
                           help='Space symbol')
        group.add_argument('--sym-blank', default='<blank>', type=str,
                           help='Blank symbol')
        return parser

    def __init__(self, idim, odim, args, flag_return=True):
        chainer.Chain.__init__(self)
        self.mtlalpha = args.mtlalpha
        assert 0 <= self.mtlalpha <= 1, "mtlalpha must be [0,1]"
        self.etype = args.etype
        self.verbose = args.verbose
        self.char_list = args.char_list
        self.outdir = args.outdir

        # below means the last number becomes eos/sos ID
        # note that sos/eos IDs are identical
        self.sos = odim - 1
        self.eos = odim - 1

        # subsample info
        # +1 means input (+1) and layers outputs (args.elayer)
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

        # label smoothing info
        if args.lsm_type:
            logging.info("Use label smoothing with " + args.lsm_type)
            labeldist = label_smoothing_dist(odim, args.lsm_type, transcript=args.train_json)
        else:
            labeldist = None

        with self.init_scope():
            # encoder
            self.enc = encoder_for(args, idim, self.subsample)
            # ctc
            self.ctc = ctc_for(args, odim)
            # attention
            self.att = att_for(args)
            # decoder
            self.dec = decoder_for(args, odim, self.sos, self.eos, self.att, labeldist)

        self.acc = None
        self.loss = None
        self.flag_return = flag_return

    def forward(self, xs, ilens, ys):
        """E2E forward

        :param xs:
        :param ilens:
        :param ys:
        :return:
        """
        # 1. encoder
        hs, ilens = self.enc(xs, ilens)

        # 3. CTC loss
        if self.mtlalpha == 0:
            loss_ctc = None
        else:
            loss_ctc = self.ctc(hs, ys)

        # 4. attention loss
        if self.mtlalpha == 1:
            loss_att = None
            acc = None
        else:
            loss_att, acc = self.dec(hs, ys)

        self.acc = acc
        alpha = self.mtlalpha
        if alpha == 0:
            self.loss = loss_att
        elif alpha == 1:
            self.loss = loss_ctc
        else:
            self.loss = alpha * loss_ctc + (1 - alpha) * loss_att

        if self.loss.data < CTC_LOSS_THRESHOLD and not math.isnan(self.loss.data):
            reporter.report({'loss_ctc': loss_ctc}, self)
            reporter.report({'loss_att': loss_att}, self)
            reporter.report({'acc': acc}, self)

            logging.info('mtl loss:' + str(self.loss.data))
            reporter.report({'loss': self.loss}, self)
        else:
            logging.warning('loss (=%f) is not correct', self.loss.data)
        if self.flag_return:
            return self.loss, loss_ctc, loss_att, acc
        else:
            return self.loss

    def recognize(self, x, recog_args, char_list, rnnlm=None):
        """E2E greedy/beam search

        :param x:
        :param recog_args:
        :param char_list:
        :param rnnlm:
        :return:
        """
        # subsample frame
        x = x[::self.subsample[0], :]
        ilen = self.xp.array(x.shape[0], dtype=np.int32)
        h = chainer.Variable(self.xp.array(x, dtype=np.float32))

        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            # 1. encoder
            # make a utt list (1) to use the same interface for encoder
            h, _ = self.enc([h], [ilen])

            # calculate log P(z_t|X) for CTC scores
            if recog_args.ctc_weight > 0.0:
                lpz = self.ctc.log_softmax(h).data[0]
            else:
                lpz = None

            # 2. decoder
            # decode the first utterance
            y = self.dec.recognize_beam(h[0], lpz, recog_args, char_list, rnnlm)

            return y

    def calculate_all_attentions(self, xs, ilens, ys):
        """E2E attention calculation

        :param xs:
        :param list xs: list of padded input sequences [(T1, idim), (T2, idim), ...]
        :param np.ndarray ilens: batch of lengths of input sequences (B)
        :param list ys: list of character id sequence tensor [(L1), (L2), (L3), ...]
        :return: attention weights (B, Lmax, Tmax)
        :rtype: float np.ndarray
        """
        hs, ilens = self.enc(xs, ilens)
        att_ws = self.dec.calculate_all_attentions(hs, ys)

        return att_ws

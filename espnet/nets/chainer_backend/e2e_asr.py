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
from espnet.nets.pytorch_backend.e2e_asr import E2E as E2E_pytorch

CTC_LOSS_THRESHOLD = 10000


class E2E(ASRInterface, chainer.Chain):
    """E2E module for chainer backend.

    Args:
        idim (int): Dimension of the inputs.
        odim (int): Dimension of the outputs.
        args (parser.args): Training config.
        flag_return (bool): If True, train() would return
            additional metrics in addition to the training
            loss.

    """
    @staticmethod
    def add_arguments(parser):
        return E2E_pytorch.add_arguments(parser)

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
        """E2E forward propagation.

        Args:
            xs (chainer.Variable): Batch of padded charactor ids. (B, Tmax)
            ilens (chainer.Variable): Batch of length of each input batch. (B,)
            ys (chainer.Variable): Batch of padded target features. (B, Lmax, odim)

        Returns:
            float: Loss that calculated by attention and ctc loss.
            float (optional): Ctc loss.
            float (optional): Attention loss.
            float (optional): Accuracy.

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
        """E2E greedy/beam search.

        Args:
            x (chainer.Variable): Input tensor for recognition.
            recog_args (parser.args): Arguments of config file.
            char_list (List[str]): List of Charactors.
            rnnlm (Module): RNNLM module defined at `espnet.lm.chainer_backend.lm`.

        Returns:
            List[Dict[str, Any]]: Result of recognition.

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
        """E2E attention calculation.

        Args:
            xs (List): List of padded input sequences. [(T1, idim), (T2, idim), ...]
            ilens (np.ndarray): Batch of lengths of input sequences. (B)
            ys (List): List of character id sequence tensor. [(L1), (L2), (L3), ...]

        Returns:
            float np.ndarray: Attention weights. (B, Lmax, Tmax)

        """
        hs, ilens = self.enc(xs, ilens)
        att_ws = self.dec.calculate_all_attentions(hs, ys)

        return att_ws

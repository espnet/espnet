#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


from __future__ import division

import logging
import math
import os

import chainer
import numpy as np
import six

from chainer import reporter
from chainer import functions as F
from chainer import links as L

from espnet.nets.e2e_asr_common import label_smoothing_dist
from espnet.nets.mt_interface import MTInterface

import espnet.nets.chainer_backend.deterministic_embed_id as DL
from espnet.nets.chainer_backend.rnn.attentions import att_for
from espnet.nets.chainer_backend.rnn.decoders import decoder_for
from espnet.nets.chainer_backend.rnn.encoders import encoder_for


class E2E(MTInterface, chainer.Chain):
    """E2E module

    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param Namespace args: argument Namespace containing options
    """

    def __init__(self, idim, odim, args):
        super(E2E, self).__init__()
        chainer.Chain.__init__(self)
        self.etype = args.etype
        self.verbose = args.verbose
        self.char_list = args.char_list
        self.outdir = args.outdir

        # below means the last number becomes eos/sos ID
        # note that sos/eos IDs are identical
        self.sos = odim - 1
        self.eos = odim - 1
        self.pad = odim

        # subsample info
        # +1 means input (+1) and layers outputs (args.elayer)
        subsample = np.ones(args.elayers + 1, dtype=np.int)
        logging.warning('Subsampling is not performed for machine translation.')
        logging.info('subsample: ' + ' '.join([str(x) for x in subsample]))
        self.subsample = subsample

        # label smoothing info
        if args.lsm_type and os.path.isfile(args.train_json):
            logging.info("Use label smoothing with " + args.lsm_type)
            labeldist = label_smoothing_dist(odim, args.lsm_type, transcript=args.train_json)
        else:
            labeldist = None

        # multilingual related
        self.replace_sos = args.replace_sos
        self.dropout = args.dropout_rate
        with self.init_scope():
            # encoder
            self.embed_src = DL.EmbedID(idim + 1, args.eunits)
            self.enc = encoder_for(args, args.eunits, self.subsample)
            # attention
            self.att = att_for(args)
            # decoder
            self.dec = decoder_for(args, odim, self.sos, self.eos, self.att, labeldist)
        self.rnnlm = None

        self.logzero = -10000000000.0
        self.loss = None
        self.acc = None

    def forward(self, xs_pad, ilens, ys_pad):
        """E2E forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, Lmax)
        :return: loss value
        :rtype: torch.Tensor
        """
        # 1. Encoder
        # xs_pad, ys_pad, tgt_lang_ids = self.target_lang_biasing_train(xs_pad, ilens, ys_pad)
        xs_pad = F.dropout(self.embed_src(xs_pad), self.dropout)
        xs_pad = [xs_pad[i] for i in range(xs_pad.shape[0])]
        hs_pad, _ = self.enc(xs_pad, ilens)

        # 3. attention loss
        loss, acc = self.dec(hs_pad, ys_pad)
        self.acc = acc
        # self.ppl = ppl

        self.loss = loss
        loss_data = float(self.loss.data)
        if not math.isnan(loss_data):
            reporter.report({'acc': acc}, self)
            logging.info('loss:' + str(self.loss.data))
            reporter.report({'loss': self.loss}, self)
        else:
            logging.warning('loss (=%f) is not correct', loss_data)
        return self.loss

    def target_lang_biasing_train(self, xs_pad, ilens, ys_pad):
        """Replace <sos> with target language IDs for multilingual MT during training.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :return: source text without language IDs
        :rtype: torch.Tensor
        :return: target text without language IDs
        :rtype: torch.Tensor
        :return: target language IDs
        :rtype: torch.Tensor (B, 1)
        """
        tgt_lang_ids = None
        if self.replace_sos:
            # remove language ID in the beggining
            tgt_lang_ids = ys_pad[:, 0].unsqueeze(1)
            xs_pad = xs_pad[:, 1:]
            ys_pad = ys_pad[:, 1:]
            ilens -= 1
        return xs_pad, ys_pad, tgt_lang_ids

    def translate(self, x, trans_args, char_list, rnnlm=None):
        """E2E beam search

        :param ndarray x: input source text feature (T, D)
        :param Namespace trans_args: argument Namespace containing options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """
        xp = self.xp
        with chainer.no_backprop_mode():
            # 1. encoder
            # make a utt list (1) to use the same interface for encoder
            if self.replace_sos:
                ilen = [len(x[0][1:])]
                h = xp.array(np.fromiter(map(int, x[0][1:]), dtype=np.int64))
            else:
                ilen = [len(x[0])]
                h = xp.array(np.fromiter(map(int, x[0]), dtype=np.int64))
            hs, _, _ = self.enc(F.dropout(self.embed_src(h.unsqueeze(0)), self.dropout), ilen)

            # 2. decoder
            # decode the first utterance
            y = self.dec.recognize_beam(hs[0], None, trans_args, char_list, rnnlm)

        return y

    def translate_batch(self, xs, trans_args, char_list, rnnlm=None):
        """E2E beam search

        :param list xs: list of input source text feature arrays [(T_1, D), (T_2, D), ...]
        :param Namespace trans_args: argument Namespace containing options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """
        prev = self.training
        self.eval()

        # 1. Encoder
        if self.replace_sos:
            ilens = np.fromiter((len(xx[1:]) for xx in xs), dtype=np.int64)
            hs = [to_device(self, torch.from_numpy(xx[1:])) for xx in xs]
        else:
            ilens = np.fromiter((len(xx) for xx in xs), dtype=np.int64)
            hs = [to_device(self, torch.from_numpy(xx)) for xx in xs]
        xpad = pad_list(hs, self.pad)
        hs_pad, hlens, _ = self.enc(self.dropout_emb_src(self.embed_src(xpad)), ilens)

        # 2. Decoder
        hlens = torch.tensor(list(map(int, hlens)))  # make sure hlens is tensor
        y = self.dec.recognize_beam_batch(hs_pad, hlens, None, trans_args, char_list, rnnlm)

        if prev:
            self.train()
        return y

    def calculate_all_attentions(self, xs_pad, ilens, ys_pad):
        """E2E attention calculation

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, Lmax)
        :return: attention weights with the following shape,
            1) multi-head case => attention weights (B, H, Lmax, Tmax),
            2) other case => attention weights (B, Lmax, Tmax).
        :rtype: float ndarray
        """
        with chainer.no_backprop_mode():
            # 1. Encoder
            xs_pad, ys_pad, tgt_lang_ids = self.target_lang_biasing_train(xs_pad, ilens, ys_pad)
            hpad, hlens, _ = self.enc(F.dropout(self.embed_src(xs_pad), self.dropout), ilens)

            # 2. Decoder
            att_ws = self.dec.calculate_all_attentions(hlens, ys_pad)

        return att_ws

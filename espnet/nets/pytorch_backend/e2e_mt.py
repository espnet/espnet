#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""RNN sequence-to-sequence text translation model (pytorch)."""

from __future__ import division

import argparse
import logging
import math
import os

import nltk

import chainer
import numpy as np
import torch

from chainer import reporter
from espnet.nets.e2e_asr_common import label_smoothing_dist
from espnet.nets.mt_interface import MTInterface
from espnet.nets.pytorch_backend.initialization import uniform_init_parameters
from espnet.nets.pytorch_backend.nets_utils import get_subsample
from espnet.nets.pytorch_backend.nets_utils import pad_list
from espnet.nets.pytorch_backend.nets_utils import to_device
from espnet.nets.pytorch_backend.rnn.attentions import att_for
from espnet.nets.pytorch_backend.rnn.decoders import decoder_for
from espnet.nets.pytorch_backend.rnn.encoders import encoder_for


class Reporter(chainer.Chain):
    """A chainer reporter wrapper."""

    def report(self, loss, acc, ppl, bleu):
        """Report at every step."""
        reporter.report({'loss': loss}, self)
        reporter.report({'acc': acc}, self)
        reporter.report({'ppl': ppl}, self)
        reporter.report({'bleu': bleu}, self)


class E2E(MTInterface, torch.nn.Module):
    """E2E module.

    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param Namespace args: argument Namespace containing options

    """

    @staticmethod
    def add_arguments(parser):
        """Add arguments."""
        E2E.encoder_add_arguments(parser)
        E2E.attention_add_arguments(parser)
        E2E.decoder_add_arguments(parser)
        return parser

    @staticmethod
    def encoder_add_arguments(parser):
        """Add arguments for the encoder."""
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
        """Add arguments for the attention."""
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
        """Add arguments for the decoder."""
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
        group.add_argument('--lsm-type', const='', default='', type=str, nargs='?',
                           choices=['', 'unigram'],
                           help='Apply label smoothing with a specified distribution type')
        return parser

    def __init__(self, idim, odim, args):
        """Construct an E2E object.

        :param int idim: dimension of inputs
        :param int odim: dimension of outputs
        :param Namespace args: argument Namespace containing options

        """
        super(E2E, self).__init__()
        torch.nn.Module.__init__(self)
        self.etype = args.etype
        self.verbose = args.verbose
        # NOTE: for self.build method
        args.char_list = getattr(args, "char_list", None)
        self.char_list = args.char_list
        self.outdir = args.outdir
        self.space = args.sym_space
        self.blank = args.sym_blank
        self.reporter = Reporter()

        # below means the last number becomes eos/sos ID
        # note that sos/eos IDs are identical
        self.sos = odim - 1
        self.eos = odim - 1
        self.pad = 0
        # NOTE: we reserve index:0 for <pad> although this is reserved for a blank class
        # in ASR. However, blank labels are not used in NMT. To keep the vocabulary size,
        # we use index:0 for padding instead of adding one more class.

        # subsample info
        self.subsample = get_subsample(args, mode='mt', arch='rnn')

        # label smoothing info
        if args.lsm_type and os.path.isfile(args.train_json):
            logging.info("Use label smoothing with " + args.lsm_type)
            labeldist = label_smoothing_dist(odim, args.lsm_type, transcript=args.train_json)
        else:
            labeldist = None

        # multilingual related
        self.multilingual = getattr(args, "multilingual", False)
        self.replace_sos = getattr(args, "replace_sos", False)

        # encoder
        self.embed = torch.nn.Embedding(idim, args.eunits, padding_idx=self.pad)
        self.dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.enc = encoder_for(args, args.eunits, self.subsample)
        # attention
        self.att = att_for(args)
        # decoder
        self.dec = decoder_for(args, odim, self.sos, self.eos, self.att, labeldist)

        # tie source and target emeddings
        if args.tie_src_tgt_embedding:
            if idim != odim:
                raise ValueError('When using tie_src_tgt_embedding, idim and odim must be equal.')
            if args.eunits != args.dunits:
                raise ValueError('When using tie_src_tgt_embedding, eunits and dunits must be equal.')
            self.embed.weight = self.dec.embed.weight

        # tie emeddings and the classfier
        if args.tie_classifier:
            if args.context_residual:
                raise ValueError('When using tie_classifier, context_residual must be turned off.')
            self.dec.output.weight = self.dec.embed.weight

        # weight initialization
        self.init_like_fairseq()

        # options for beam search
        if args.report_bleu:
            trans_args = {'beam_size': args.beam_size, 'penalty': args.penalty,
                          'ctc_weight': 0, 'maxlenratio': args.maxlenratio,
                          'minlenratio': args.minlenratio, 'lm_weight': args.lm_weight,
                          'rnnlm': args.rnnlm, 'nbest': args.nbest,
                          'space': args.sym_space, 'blank': args.sym_blank,
                          'tgt_lang': False}

            self.trans_args = argparse.Namespace(**trans_args)
            self.report_bleu = args.report_bleu
        else:
            self.report_bleu = False
        self.rnnlm = None

        self.logzero = -10000000000.0
        self.loss = None
        self.acc = None

    def init_like_fairseq(self):
        """Initialize weight like fairseq.

        fairseq basically uses W, b, EmbedID.W ~ Uniform(-0.1, 0.1),
        """
        uniform_init_parameters(self)
        # exceptions
        # embed weight ~ Normal(-0.1, 0.1)
        torch.nn.init.uniform_(self.embed.weight, -0.1, 0.1)
        torch.nn.init.constant_(self.embed.weight[self.pad], 0)
        torch.nn.init.uniform_(self.dec.embed.weight, -0.1, 0.1)
        torch.nn.init.constant_(self.dec.embed.weight[self.pad], 0)

    def forward(self, xs_pad, ilens, ys_pad):
        """E2E forward.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, Lmax)
        :return: loss value
        :rtype: torch.Tensor
        """
        # 1. Encoder
        xs_pad, ys_pad = self.target_language_biasing(xs_pad, ilens, ys_pad)
        hs_pad, hlens, _ = self.enc(self.dropout(self.embed(xs_pad)), ilens)

        # 3. attention loss
        loss, acc, ppl = self.dec(hs_pad, hlens, ys_pad)
        self.acc = acc
        self.ppl = ppl

        # 5. compute bleu
        if self.training or not self.report_bleu:
            bleu = 0.0
        else:
            lpz = None

            bleus = []
            nbest_hyps = self.dec.recognize_beam_batch(
                hs_pad, torch.tensor(hlens), lpz,
                self.trans_args, self.char_list,
                self.rnnlm)
            # remove <sos> and <eos>
            y_hats = [nbest_hyp[0]['yseq'][1:-1] for nbest_hyp in nbest_hyps]
            for i, y_hat in enumerate(y_hats):
                y_true = ys_pad[i]

                seq_hat = [self.char_list[int(idx)] for idx in y_hat if int(idx) != -1]
                seq_true = [self.char_list[int(idx)] for idx in y_true if int(idx) != -1]
                seq_hat_text = "".join(seq_hat).replace(self.trans_args.space, ' ')
                seq_hat_text = seq_hat_text.replace(self.trans_args.blank, '')
                seq_true_text = "".join(seq_true).replace(self.trans_args.space, ' ')

                bleu = nltk.bleu_score.sentence_bleu([seq_true_text], seq_hat_text) * 100
                bleus.append(bleu)

            bleu = 0.0 if not self.report_bleu else sum(bleus) / len(bleus)

        self.loss = loss

        loss_data = float(self.loss)
        if not math.isnan(loss_data):
            self.reporter.report(loss_data, acc, ppl, bleu)
        else:
            logging.warning('loss (=%f) is not correct', loss_data)
        return self.loss

    def target_language_biasing(self, xs_pad, ilens, ys_pad):
        """Prepend target language IDs to source sentences for multilingual NMT.

        These tags are prepended in source/target sentences as pre-processing.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :return: source text without language IDs
        :rtype: torch.Tensor
        :return: target text without language IDs
        :rtype: torch.Tensor
        :return: target language IDs
        :rtype: torch.Tensor (B, 1)
        """
        if self.multilingual:
            # remove language ID in the beggining
            tgt_lang_ids = ys_pad[:, 0].unsqueeze(1)
            xs_pad = xs_pad[:, 1:]  # remove source language IDs here
            ys_pad = ys_pad[:, 1:]

            # prepend target language ID to source sentences
            xs_pad = torch.cat([tgt_lang_ids, xs_pad], dim=1)
        return xs_pad, ys_pad

    def translate(self, x, trans_args, char_list, rnnlm=None):
        """E2E beam search.

        :param ndarray x: input source text feature (B, T, D)
        :param Namespace trans_args: argument Namespace containing options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """
        prev = self.training
        self.eval()

        # 1. encoder
        # make a utt list (1) to use the same interface for encoder
        if self.multilingual:
            ilen = [len(x[0][1:])]
            h = to_device(self, torch.from_numpy(np.fromiter(map(int, x[0][1:]), dtype=np.int64)))
        else:
            ilen = [len(x[0])]
            h = to_device(self, torch.from_numpy(np.fromiter(map(int, x[0]), dtype=np.int64)))
        hs, _, _ = self.enc(self.dropout(self.embed(h.unsqueeze(0))), ilen)

        # 2. decoder
        # decode the first utterance
        y = self.dec.recognize_beam(hs[0], None, trans_args, char_list, rnnlm)

        if prev:
            self.train()
        return y

    def translate_batch(self, xs, trans_args, char_list, rnnlm=None):
        """E2E beam search.

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
        if self.multilingual:
            ilens = np.fromiter((len(xx[1:]) for xx in xs), dtype=np.int64)
            hs = [to_device(self, torch.from_numpy(xx[1:])) for xx in xs]
        else:
            ilens = np.fromiter((len(xx) for xx in xs), dtype=np.int64)
            hs = [to_device(self, torch.from_numpy(xx)) for xx in xs]
        xpad = pad_list(hs, self.pad)
        hs_pad, hlens, _ = self.enc(self.dropout(self.embed(xpad)), ilens)

        # 2. Decoder
        hlens = torch.tensor(list(map(int, hlens)))  # make sure hlens is tensor
        y = self.dec.recognize_beam_batch(hs_pad, hlens, None, trans_args, char_list, rnnlm)

        if prev:
            self.train()
        return y

    def calculate_all_attentions(self, xs_pad, ilens, ys_pad):
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
            # 1. Encoder
            xs_pad, ys_pad = self.target_language_biasing(xs_pad, ilens, ys_pad)
            hpad, hlens, _ = self.enc(self.dropout(self.embed(xs_pad)), ilens)

            # 2. Decoder
            att_ws = self.dec.calculate_all_attentions(hpad, hlens, ys_pad)

        return att_ws

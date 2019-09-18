#!/usr/bin/env python
# encoding: utf-8

# Copyright 2019 Nankai University (Chengyi Wang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""RNN Tandem Connectionist Encoding Network (pytorch)."""

from __future__ import division

import argparse
import copy
import logging
import math
import os

import editdistance
import nltk

import chainer
import numpy as np
import six
import torch

from chainer import reporter
from espnet.nets.e2e_asr_common import label_smoothing_dist
from espnet.nets.pytorch_backend.initialization import lecun_normal_init_parameters
from espnet.nets.pytorch_backend.initialization import set_forget_bias_to_one
from espnet.nets.pytorch_backend.nets_utils import pad_list
from espnet.nets.pytorch_backend.nets_utils import to_device
from espnet.nets.pytorch_backend.nets_utils import to_torch_tensor
from espnet.nets.pytorch_backend.rnn.attentions import att_for
from espnet.nets.pytorch_backend.rnn.decoders import decoder_for
from espnet.nets.pytorch_backend.rnn.encoders import Encoder
from espnet.nets.pytorch_backend.ctc import ctc_for
from espnet.nets.st_interface import STInterface

CTC_LOSS_THRESHOLD = 10000

class Reporter(chainer.Chain):
    """A chainer reporter wrapper."""

    def report(self, loss_ctc, loss_st, loss_mt, acc, bleu):
        """Report at every step."""
        reporter.report({'loss_ctc': loss_ctc}, self)
        reporter.report({'loss_st': loss_st}, self)
        reporter.report({'loss_mt': loss_mt}, self)
        reporter.report({'acc': acc}, self)
        reporter.report({'bleu': bleu}, self)


class E2E(STInterface, torch.nn.Module):
    """E2E module.

    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param Namespace args: argument Namespace containing options
    :param E2E (ASRInterface) asr_model: pre-trained ASR model for encoder initialization
    :param E2E (MTInterface) mt_model: pre-trained NMT model for decoder initialization

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
        group.add_argument('--setype', default='blstmp', type=str,
                           choices=['lstm', 'blstm', 'lstmp', 'blstmp', 'vgglstmp', 'vggblstmp', 'vgglstm', 'vggblstm',
                                    'gru', 'bgru', 'grup', 'bgrup', 'vgggrup', 'vggbgrup', 'vgggru', 'vggbgru'],
                           help='Type of speech encoder network architecture')
        group.add_argument('--selayers', default=4, type=int,
                           help='Number of speech encoder layers (for shared recognition part in multi-speaker asr mode)')
        group.add_argument('--tetype', default='blstm', type=str,
                           choices=['lstm', 'blstm', 'lstmp', 'blstmp', 'vgglstmp', 'vggblstmp', 'vgglstm', 'vggblstm',
                                    'gru', 'bgru', 'grup', 'bgrup', 'vgggrup', 'vggbgrup', 'vgggru', 'vggbgru'],
                            help='Type of text encoder network architecture')
        group.add_argument('--telayers', default=2, type=int,
                           help='Number of text encoder layers')
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
        return parser

    def __init__(self, idim, odim, mdim, args, asr_model=None, mt_model=None, share_weight=True):
        """Construct an E2E object."""
        super(E2E, self).__init__()
        torch.nn.Module.__init__(self)
        self.setype = args.setype
        self.tetype = args.tetype
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

        # subsample info
        # +1 means input (+1) and layers outputs (args.elayer)
        subsample = np.ones(args.selayers + 1, dtype=np.int)
        if args.setype.endswith("p") and not args.setype.startswith("vgg"):
            ss = args.subsample.split("_")
            for j in range(min(args.selayers + 1, len(ss))):
                subsample[j] = int(ss[j])
        else:
            logging.warning(
                'Subsampling is not performed for vgg*. It is performed in max pooling layers at CNN.')
        logging.info('subsample: ' + ' '.join([str(x) for x in subsample]))
        self.subsample = subsample

        # label smoothing info
        if args.lsm_type and os.path.isfile(args.train_json):
            logging.info("Use label smoothing with " + args.lsm_type)
            labeldist = label_smoothing_dist(odim, args.lsm_type, transcript=args.train_json)
        else:
            labeldist = None

        # multilingual related
        self.multilingual = getattr(args, "multilingual", False)
        self.replace_sos = args.replace_sos

        # speech encoder
        self.senc = Encoder(args.setype, idim, args.selayers, args.eunits, args.eprojs, subsample, args.dropout_rate)

        # text encoder
        self.tenc = Encoder(args.tetype, args.eprojs, args.telayers, args.eunits, args.eprojs, subsample, args.dropout_rate)

        # source embedding
        self.embed = torch.nn.Embedding(mdim, args.eprojs, padding_idx=mdim-1)
        self.dropout = torch.nn.Dropout(p=args.dropout_rate)
        # ctc
        self.ctc = ctc_for(args, mdim)

        # attention (st)
        self.att = att_for(args)
        # decoder (st)
        self.dec = decoder_for(args, odim, self.sos, self.eos, self.att, labeldist)

        # weight initialization
        self.init_like_chainer()

        # initialize speech encoder from pretrained asr model
        if asr_model:
            param_dict = dict(asr_model.named_parameters())
            for n, p in self.named_parameters():
                asr_n = n.replace('senc', 'enc')
                if 'enc.enc' in asr_n or 'ctc' in asr_n:
                    if asr_n in param_dict.keys() and p.size() == param_dict[asr_n].size():
                        p.data = param_dict[asr_n].data
                        logging.warning('Overwrite %s from asr model' % n)

        # initialize text encoder and decoder from pretrained mt model
        if mt_model:
            param_dict = dict(mt_model.named_parameters())
            for n, p in self.named_parameters():
                mt_n = n.replace('tenc', 'enc')
                if 'enc.enc' in mt_n or 'embed' in mt_n or 'dec' in mt_n:
                    if mt_n in param_dict.keys() and p.size() == param_dict[mt_n].size():
                        p.data = param_dict[mt_n].data
                        logging.warning('Overwrite %s from mt model' % n) 

        # share weights between ctc layer and source embedding layer
        if share_weight:
            self.ctc.ctc_lo.weight = self.embed.weight
        # options for beam search
        if args.report_cer or args.report_wer:
            recog_args = {'beam_size': args.beam_size, 'penalty': args.penalty,
                          'ctc_weight': 0, 'maxlenratio': args.maxlenratio,
                          'minlenratio': args.minlenratio, 'lm_weight': args.lm_weight,
                          'rnnlm': args.rnnlm, 'nbest': args.nbest,
                          'space': args.sym_space, 'blank': args.sym_blank,
                          'tgt_lang': False}

            self.recog_args = argparse.Namespace(**recog_args)
            self.report_cer = args.report_cer
            self.report_wer = args.report_wer
        else:
            self.report_cer = False
            self.report_wer = False
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
        self.loss_ctc = None
        self.loss_mt = None
        self.loss_st = None
        self.acc = None

    def init_like_chainer(self):
        """Initialize weight like chainer.

        chainer basically uses LeCun way: W ~ Normal(0, fan_in ** -0.5), b = 0
        pytorch basically uses W, b ~ Uniform(-fan_in**-0.5, fan_in**-0.5)
        however, there are two exceptions as far as I know.
        - EmbedID.W ~ Normal(0, 1)
        - LSTM.upward.b[forget_gate_range] = 1 (but not used in NStepLSTM)
        """
        lecun_normal_init_parameters(self)
        # exceptions
        # embed weight ~ Normal(0, 1)
        self.dec.embed.weight.data.normal_(0, 1)
        # forget-bias = 1.0
        # https://discuss.pytorch.org/t/set-forget-gate-bias-of-lstm/1745
        for l in six.moves.range(len(self.dec.decoder)):
            set_forget_bias_to_one(self.dec.decoder[l].bias_ih)

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


    def forward(self, xs_pad, ilens, ys_pad, task='st'):
        """E2E forward.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, Lmax)
        :return: loss value
        :rtype: torch.Tensor
        """
        # 0. Extract target language ID
        if self.multilingual:
            tgt_lang_ids = ys_pad[:, 0:1]
            ys_pad = ys_pad[:, 1:]  # remove target language ID in the beggining
        else:
            tgt_lang_ids = None


        # ST task
        if task == "st":
            hs_pad, hlens, _ = self.senc(xs_pad, ilens)
            hs_pad, hlens, _ = self.tenc(hs_pad, hlens)
            self.loss_st, acc, _ = self.dec(hs_pad, hlens, ys_pad, tgt_lang_ids=tgt_lang_ids)
            self.acc = acc
        elif task == "asr":
            hs_pad, hlens, _ = self.senc(xs_pad, ilens)
            self.loss_ctc = self.ctc(hs_pad, hlens, ys_pad)
        elif task == "mt":
            xs_pad, ys_pad = self.target_language_biasing(xs_pad, ilens, ys_pad)
            hs_pad, hlens, _ = self.tenc(self.dropout(self.embed(xs_pad)), ilens)
            self.loss_mt, acc, _ = self.dec(hs_pad, hlens, ys_pad)
        else:
            raise ValueError('Task must be one of asr, st or mt, got %s instead' % task)
        

        # 5. compute bleu
        if self.training or not self.report_bleu:
            bleu = 0.0
        else:
            lpz = None

            bleus = []
            nbest_hyps = self.dec.recognize_beam_batch(
                hs_pad, torch.tensor(hlens), lpz,
                self.trans_args, self.char_list,
                self.rnnlm,
                tgt_lang_ids=tgt_lang_ids.squeeze(1).tolist() if self.multilingual else None)
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


        if task == "st":
            self.loss = self.loss_st
            loss_st_data = float(self.loss_st)
            loss_ctc_data = None
            loss_mt_data = None
        elif task == "asr":
            self.loss = self.loss_ctc
            loss_st_data = None
            loss_ctc_data = float(self.loss_ctc)
            loss_mt_data = None
        else:
            self.loss = self.loss_mt
            loss_st_data = None
            loss_ctc_data = None
            loss_mt_data = float(self.loss_mt) 


        loss_data = float(self.loss)
        if not math.isnan(loss_data):
            self.reporter.report(loss_ctc_data, loss_st_data, loss_mt_data, self.acc, bleu)
        else:
            logging.warning('loss (=%f) is not correct', loss_data)
        return self.loss

    def scorers(self):
        """Scorers."""
        return dict(decoder=self.dec)

    def encode(self, x):
        """Encode acoustic features.

        :param ndarray x: input acoustic feature (T, D)
        :return: encoder outputs
        :rtype: torch.Tensor
        """
        self.eval()
        ilens = [x.shape[0]]

        # subsample frame
        x = x[::self.subsample[0], :]
        p = next(self.parameters())
        h = torch.as_tensor(x, device=p.device, dtype=p.dtype)
        # make a utt list (1) to use the same interface for encoder
        hs = h.contiguous().unsqueeze(0)

        # 1. encoder
        hs, hlens, _ = self.senc(hs, ilens)
        hs, _, _ = self.tenc(hs, hlens) 
        return hs.squeeze(0)

    def translate(self, x, trans_args, char_list, rnnlm=None):
        """E2E beam search.

        :param ndarray x: input acoustic feature (T, D)
        :param Namespace trans_args: argument Namespace containing options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """
        hs = self.encode(x).unsqueeze(0)
        lpz = None

        # 2. Decoder
        # decode the first utterance
        y = self.dec.recognize_beam(hs[0], lpz, trans_args, char_list, rnnlm)
        return y

    def translate_batch(self, xs, trans_args, char_list, rnnlm=None):
        """E2E beam search.

        :param list xs: list of input acoustic feature arrays [(T_1, D), (T_2, D), ...]
        :param Namespace trans_args: argument Namespace containing options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """
        prev = self.training
        self.eval()
        ilens = np.fromiter((xx.shape[0] for xx in xs), dtype=np.int64)

        # subsample frame
        xs = [xx[::self.subsample[0], :] for xx in xs]
        xs = [to_device(self, to_torch_tensor(xx).float()) for xx in xs]
        xs_pad = pad_list(xs, 0.0)

        # 1. Encoder
        hs_pad, hlens, _ = self.enc(xs_pad, ilens)
        lpz = None

        # 2. Decoder
        hlens = torch.tensor(list(map(int, hlens)))  # make sure hlens is tensor
        y = self.dec.recognize_beam_batch(hs_pad, hlens, lpz, trans_args, char_list, rnnlm)

        if prev:
            self.train()
        return y

    def calculate_all_attentions(self, xs_pad, ilens, ys_pad, ys_pad_asr):
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
            if self.multilingual:
                tgt_lang_ids = ys_pad[:, 0:1]
                ys_pad = ys_pad[:, 1:]  # remove target language ID in the beggining
            else:
                tgt_lang_ids = None
            hpad, hlens, _ = self.enc(xs_pad, ilens)

            # 2. Decoder
            att_ws = self.dec.calculate_all_attentions(hpad, hlens, ys_pad, tgt_lang_ids=tgt_lang_ids)

        return att_ws

    def subsample_frames(self, x):
        """Subsample speeh frames in the encoder."""
        # subsample frame
        x = x[::self.subsample[0], :]
        ilen = [x.shape[0]]
        h = to_device(self, torch.from_numpy(
            np.array(x, dtype=np.float32)))
        h.contiguous()
        return h, ilen

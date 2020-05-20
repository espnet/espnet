#!/usr/bin/env python
# encoding: utf-8

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""RNN sequence-to-sequence speech translation model (pytorch)."""

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

from itertools import groupby

from chainer import reporter

from espnet.nets.e2e_asr_common import label_smoothing_dist
from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.initialization import lecun_normal_init_parameters
from espnet.nets.pytorch_backend.initialization import set_forget_bias_to_one
from espnet.nets.pytorch_backend.nets_utils import get_subsample
from espnet.nets.pytorch_backend.nets_utils import pad_list
from espnet.nets.pytorch_backend.nets_utils import to_device
from espnet.nets.pytorch_backend.nets_utils import to_torch_tensor
from espnet.nets.pytorch_backend.rnn.attentions import att_for
from espnet.nets.pytorch_backend.rnn.decoders import decoder_for
from espnet.nets.pytorch_backend.rnn.encoders import encoder_for
from espnet.nets.st_interface import STInterface

CTC_LOSS_THRESHOLD = 10000


class Reporter(chainer.Chain):
    """A chainer reporter wrapper."""

    def report(
        self,
        loss_asr,
        loss_mt,
        loss_st,
        acc_asr,
        acc_mt,
        acc,
        cer_ctc,
        cer,
        wer,
        bleu,
        mtl_loss,
    ):
        """Report at every step."""
        reporter.report({"loss_asr": loss_asr}, self)
        reporter.report({"loss_mt": loss_mt}, self)
        reporter.report({"loss_st": loss_st}, self)
        reporter.report({"acc_asr": acc_asr}, self)
        reporter.report({"acc_mt": acc_mt}, self)
        reporter.report({"acc": acc}, self)
        reporter.report({"cer_ctc": cer_ctc}, self)
        reporter.report({"cer": cer}, self)
        reporter.report({"wer": wer}, self)
        reporter.report({"bleu": bleu}, self)
        logging.info("mtl loss:" + str(mtl_loss))
        reporter.report({"loss": mtl_loss}, self)


class E2E(STInterface, torch.nn.Module):
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
        group.add_argument(
            "--etype",
            default="blstmp",
            type=str,
            choices=[
                "lstm",
                "blstm",
                "lstmp",
                "blstmp",
                "vgglstmp",
                "vggblstmp",
                "vgglstm",
                "vggblstm",
                "gru",
                "bgru",
                "grup",
                "bgrup",
                "vgggrup",
                "vggbgrup",
                "vgggru",
                "vggbgru",
            ],
            help="Type of encoder network architecture",
        )
        group.add_argument(
            "--elayers",
            default=4,
            type=int,
            help="Number of encoder layers (for shared recognition part "
            "in multi-speaker asr mode)",
        )
        group.add_argument(
            "--eunits",
            "-u",
            default=300,
            type=int,
            help="Number of encoder hidden units",
        )
        group.add_argument(
            "--eprojs", default=320, type=int, help="Number of encoder projection units"
        )
        group.add_argument(
            "--subsample",
            default="1",
            type=str,
            help="Subsample input frames x_y_z means "
            "subsample every x frame at 1st layer, "
            "every y frame at 2nd layer etc.",
        )
        return parser

    @staticmethod
    def attention_add_arguments(parser):
        """Add arguments for the attention."""
        group = parser.add_argument_group("E2E attention setting")
        # attention
        group.add_argument(
            "--atype",
            default="dot",
            type=str,
            choices=[
                "noatt",
                "dot",
                "add",
                "location",
                "coverage",
                "coverage_location",
                "location2d",
                "location_recurrent",
                "multi_head_dot",
                "multi_head_add",
                "multi_head_loc",
                "multi_head_multi_res_loc",
            ],
            help="Type of attention architecture",
        )
        group.add_argument(
            "--adim",
            default=320,
            type=int,
            help="Number of attention transformation dimensions",
        )
        group.add_argument(
            "--awin", default=5, type=int, help="Window size for location2d attention"
        )
        group.add_argument(
            "--aheads",
            default=4,
            type=int,
            help="Number of heads for multi head attention",
        )
        group.add_argument(
            "--aconv-chans",
            default=-1,
            type=int,
            help="Number of attention convolution channels \
                           (negative value indicates no location-aware attention)",
        )
        group.add_argument(
            "--aconv-filts",
            default=100,
            type=int,
            help="Number of attention convolution filters \
                           (negative value indicates no location-aware attention)",
        )
        group.add_argument(
            "--dropout-rate",
            default=0.0,
            type=float,
            help="Dropout rate for the encoder",
        )
        return parser

    @staticmethod
    def decoder_add_arguments(parser):
        """Add arguments for the decoder."""
        group = parser.add_argument_group("E2E encoder setting")
        group.add_argument(
            "--dtype",
            default="lstm",
            type=str,
            choices=["lstm", "gru"],
            help="Type of decoder network architecture",
        )
        group.add_argument(
            "--dlayers", default=1, type=int, help="Number of decoder layers"
        )
        group.add_argument(
            "--dunits", default=320, type=int, help="Number of decoder hidden units"
        )
        group.add_argument(
            "--dropout-rate-decoder",
            default=0.0,
            type=float,
            help="Dropout rate for the decoder",
        )
        group.add_argument(
            "--sampling-probability",
            default=0.0,
            type=float,
            help="Ratio of predicted labels fed back to decoder",
        )
        group.add_argument(
            "--lsm-type",
            const="",
            default="",
            type=str,
            nargs="?",
            choices=["", "unigram"],
            help="Apply label smoothing with a specified distribution type",
        )
        return parser

    def __init__(self, idim, odim, args):
        """Construct an E2E object.

        :param int idim: dimension of inputs
        :param int odim: dimension of outputs
        :param Namespace args: argument Namespace containing options
        """
        super(E2E, self).__init__()
        torch.nn.Module.__init__(self)
        self.asr_weight = getattr(args, "asr_weight", 0)
        self.mt_weight = getattr(args, "mt_weight", 0)
        self.mtlalpha = args.mtlalpha
        assert 0.0 <= self.asr_weight < 1.0, "asr_weight should be [0.0, 1.0)"
        assert 0.0 <= self.mt_weight < 1.0, "mt_weight should be [0.0, 1.0)"
        assert 0.0 <= self.mtlalpha <= 1.0, "mtlalpha should be [0.0, 1.0]"
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
        # in ASR. However, blank labels are not used in NMT.
        # To keep the vocabulary size,
        # we use index:0 for padding instead of adding one more class.

        # subsample info
        self.subsample = get_subsample(args, mode="st", arch="rnn")

        # label smoothing info
        if args.lsm_type and os.path.isfile(args.train_json):
            logging.info("Use label smoothing with " + args.lsm_type)
            labeldist = label_smoothing_dist(
                odim, args.lsm_type, transcript=args.train_json
            )
        else:
            labeldist = None

        # multilingual E2E-ST related
        self.multilingual = getattr(args, "multilingual", False)
        self.joint_asr = getattr(args, "joint_asr", False)
        self.replace_sos = getattr(args, "replace_sos", False)

        # encoder
        self.enc = encoder_for(args, idim, self.subsample)
        # attention (ST)
        self.att = att_for(args)
        # decoder (ST)
        self.dec = decoder_for(args, odim, self.sos, self.eos, self.att, labeldist)

        # submodule for ASR task
        self.ctc = None
        self.att_asr = None
        self.dec_asr = None
        if self.asr_weight > 0:
            if self.mtlalpha > 0.0:
                self.ctc = CTC(
                    odim,
                    args.eprojs,
                    args.dropout_rate,
                    ctc_type=args.ctc_type,
                    reduce=True,
                )
            if self.mtlalpha < 1.0:
                # attention (asr)
                self.att_asr = att_for(args)
                # decoder (asr)
                args_asr = copy.deepcopy(args)
                args_asr.atype = "location"  # TODO(hirofumi0810): make this option
                self.dec_asr = decoder_for(
                    args_asr, odim, self.sos, self.eos, self.att_asr, labeldist
                )

        # submodule for MT task
        if self.mt_weight > 0:
            self.embed_mt = torch.nn.Embedding(odim, args.eunits, padding_idx=self.pad)
            self.dropout_mt = torch.nn.Dropout(p=args.dropout_rate)
            self.enc_mt = encoder_for(
                args, args.eunits, subsample=np.ones(args.elayers + 1, dtype=np.int)
            )

        # weight initialization
        self.init_like_chainer()

        # options for beam search
        if self.asr_weight > 0 and args.report_cer or args.report_wer:
            recog_args = {
                "beam_size": args.beam_size,
                "penalty": args.penalty,
                "ctc_weight": args.ctc_weight,
                "maxlenratio": args.maxlenratio,
                "minlenratio": args.minlenratio,
                "lm_weight": args.lm_weight,
                "rnnlm": args.rnnlm,
                "nbest": args.nbest,
                "space": args.sym_space,
                "blank": args.sym_blank,
                "tgt_lang": False,
            }

            self.recog_args = argparse.Namespace(**recog_args)
            self.report_cer = args.report_cer
            self.report_wer = args.report_wer
        else:
            self.report_cer = False
            self.report_wer = False
        if args.report_bleu:
            trans_args = {
                "beam_size": args.beam_size,
                "penalty": args.penalty,
                "ctc_weight": 0,
                "maxlenratio": args.maxlenratio,
                "minlenratio": args.minlenratio,
                "lm_weight": args.lm_weight,
                "rnnlm": args.rnnlm,
                "nbest": args.nbest,
                "space": args.sym_space,
                "blank": args.sym_blank,
                "tgt_lang": False,
            }

            self.trans_args = argparse.Namespace(**trans_args)
            self.report_bleu = args.report_bleu
        else:
            self.report_bleu = False
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
        lecun_normal_init_parameters(self)
        # exceptions
        # embed weight ~ Normal(0, 1)
        self.dec.embed.weight.data.normal_(0, 1)
        # forget-bias = 1.0
        # https://discuss.pytorch.org/t/set-forget-gate-bias-of-lstm/1745
        for i in six.moves.range(len(self.dec.decoder)):
            set_forget_bias_to_one(self.dec.decoder[i].bias_ih)

    def forward(self, xs_pad, ilens, ys_pad, ys_pad_src):
        """E2E forward.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad:
            batch of padded character id sequence tensor (B, Lmax)
        :return: loss value
        :rtype: torch.Tensor
        """
        # 0. Extract target language ID
        if self.multilingual:
            tgt_lang_ids = ys_pad[:, 0:1]
            ys_pad = ys_pad[:, 1:]  # remove target language ID in the beggining
        else:
            tgt_lang_ids = None

        # 1. Encoder
        hs_pad, hlens, _ = self.enc(xs_pad, ilens)

        # 2. ST attention loss
        self.loss_st, acc, _ = self.dec(hs_pad, hlens, ys_pad, lang_ids=tgt_lang_ids)
        self.acc = acc

        # 2. ASR CTC loss
        if self.asr_weight == 0 or self.mtlalpha == 0:
            self.loss_ctc = 0.0
        else:
            self.loss_ctc = self.ctc(hs_pad, hlens, ys_pad_src)

        # 3. ASR attention loss
        if self.asr_weight == 0 or self.mtlalpha == 1:
            self.loss_asr = 0.0
            self.acc_asr = 0.0
        else:
            self.loss_asr, acc_asr, _ = self.dec_asr(hs_pad, hlens, ys_pad_src)
            self.acc_asr = acc_asr

        # 3. MT attention loss
        if self.mt_weight == 0:
            self.loss_mt = 0.0
            self.acc_mt = 0.0
        else:
            # ys_pad_src, ys_pad = self.target_forcing(ys_pad_src, ys_pad)
            ilens_mt = torch.sum(ys_pad_src != -1, dim=1).cpu().numpy()
            # NOTE: ys_pad_src is padded with -1
            ys_src = [y[y != -1] for y in ys_pad_src]  # parse padded ys_src
            ys_zero_pad_src = pad_list(ys_src, self.pad)  # re-pad with zero
            hs_pad_mt, hlens_mt, _ = self.enc_mt(
                self.dropout_mt(self.embed_mt(ys_zero_pad_src)), ilens_mt
            )
            self.loss_mt, acc_mt, _ = self.dec(hs_pad_mt, hlens_mt, ys_pad)
            self.acc_mt = acc_mt

        # 4. compute cer without beam search
        if (self.asr_weight == 0 or self.mtlalpha == 0) or self.char_list is None:
            cer_ctc = None
        else:
            cers = []

            y_hats = self.ctc.argmax(hs_pad).data
            for i, y in enumerate(y_hats):
                y_hat = [x[0] for x in groupby(y)]
                y_true = ys_pad_src[i]

                seq_hat = [self.char_list[int(idx)] for idx in y_hat if int(idx) != -1]
                seq_true = [
                    self.char_list[int(idx)] for idx in y_true if int(idx) != -1
                ]
                seq_hat_text = "".join(seq_hat).replace(self.space, " ")
                seq_hat_text = seq_hat_text.replace(self.blank, "")
                seq_true_text = "".join(seq_true).replace(self.space, " ")

                hyp_chars = seq_hat_text.replace(" ", "")
                ref_chars = seq_true_text.replace(" ", "")
                if len(ref_chars) > 0:
                    cers.append(
                        editdistance.eval(hyp_chars, ref_chars) / len(ref_chars)
                    )

            cer_ctc = sum(cers) / len(cers) if cers else None

        # 5. compute cer/wer
        if self.training or (
            self.asr_weight == 0
            or self.mtlalpha == 1
            or not (self.report_cer or self.report_wer)
        ):
            cer, wer = 0.0, 0.0
            # oracle_cer, oracle_wer = 0.0, 0.0
        else:
            if (
                self.asr_weight > 0 and self.mtlalpha > 0
            ) and self.recog_args.ctc_weight > 0.0:
                lpz = self.ctc.log_softmax(hs_pad).data
            else:
                lpz = None

            word_eds, word_ref_lens, char_eds, char_ref_lens = [], [], [], []
            nbest_hyps_asr = self.dec_asr.recognize_beam_batch(
                hs_pad,
                torch.tensor(hlens),
                lpz,
                self.recog_args,
                self.char_list,
                self.rnnlm,
            )
            # remove <sos> and <eos>
            y_hats = [nbest_hyp[0]["yseq"][1:-1] for nbest_hyp in nbest_hyps_asr]
            for i, y_hat in enumerate(y_hats):
                y_true = ys_pad[i]

                seq_hat = [self.char_list[int(idx)] for idx in y_hat if int(idx) != -1]
                seq_true = [
                    self.char_list[int(idx)] for idx in y_true if int(idx) != -1
                ]
                seq_hat_text = "".join(seq_hat).replace(self.recog_args.space, " ")
                seq_hat_text = seq_hat_text.replace(self.recog_args.blank, "")
                seq_true_text = "".join(seq_true).replace(self.recog_args.space, " ")

                hyp_words = seq_hat_text.split()
                ref_words = seq_true_text.split()
                word_eds.append(editdistance.eval(hyp_words, ref_words))
                word_ref_lens.append(len(ref_words))
                hyp_chars = seq_hat_text.replace(" ", "")
                ref_chars = seq_true_text.replace(" ", "")
                char_eds.append(editdistance.eval(hyp_chars, ref_chars))
                char_ref_lens.append(len(ref_chars))

            wer = (
                0.0
                if not self.report_wer
                else float(sum(word_eds)) / sum(word_ref_lens)
            )
            cer = (
                0.0
                if not self.report_cer
                else float(sum(char_eds)) / sum(char_ref_lens)
            )

        # 6. compute bleu
        if self.training or not self.report_bleu:
            bleu = 0.0
        else:
            lpz = None

            bleus = []
            nbest_hyps = self.dec.recognize_beam_batch(
                hs_pad,
                torch.tensor(hlens),
                lpz,
                self.trans_args,
                self.char_list,
                self.rnnlm,
                lang_ids=tgt_lang_ids.squeeze(1).tolist()
                if self.multilingual
                else None,
            )
            # remove <sos> and <eos>
            y_hats = [nbest_hyp[0]["yseq"][1:-1] for nbest_hyp in nbest_hyps]
            for i, y_hat in enumerate(y_hats):
                y_true = ys_pad[i]

                seq_hat = [self.char_list[int(idx)] for idx in y_hat if int(idx) != -1]
                seq_true = [
                    self.char_list[int(idx)] for idx in y_true if int(idx) != -1
                ]
                seq_hat_text = "".join(seq_hat).replace(self.trans_args.space, " ")
                seq_hat_text = seq_hat_text.replace(self.trans_args.blank, "")
                seq_true_text = "".join(seq_true).replace(self.trans_args.space, " ")

                bleu = (
                    nltk.bleu_score.sentence_bleu([seq_true_text], seq_hat_text) * 100
                )
                bleus.append(bleu)

            bleu = 0.0 if not self.report_bleu else sum(bleus) / len(bleus)

        alpha = self.mtlalpha
        self.loss = (
            (1 - self.asr_weight - self.mt_weight) * self.loss_st
            + self.asr_weight * (alpha * self.loss_ctc + (1 - alpha) * self.loss_asr)
            + self.mt_weight * self.loss_mt
        )
        loss_st_data = float(self.loss_st)
        loss_asr_data = float(alpha * self.loss_ctc + (1 - alpha) * self.loss_asr)
        loss_mt_data = None if self.mt_weight == 0 else float(self.loss_mt)

        loss_data = float(self.loss)
        if loss_data < CTC_LOSS_THRESHOLD and not math.isnan(loss_data):
            self.reporter.report(
                loss_asr_data,
                loss_mt_data,
                loss_st_data,
                self.acc_asr,
                self.acc_mt,
                acc,
                cer_ctc,
                cer,
                wer,
                bleu,
                loss_data,
            )
        else:
            logging.warning("loss (=%f) is not correct", loss_data)
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
        x = x[:: self.subsample[0], :]
        p = next(self.parameters())
        h = torch.as_tensor(x, device=p.device, dtype=p.dtype)
        # make a utt list (1) to use the same interface for encoder
        hs = h.contiguous().unsqueeze(0)

        # 1. encoder
        hs, _, _ = self.enc(hs, ilens)
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
        xs = [xx[:: self.subsample[0], :] for xx in xs]
        xs = [to_device(self, to_torch_tensor(xx).float()) for xx in xs]
        xs_pad = pad_list(xs, 0.0)

        # 1. Encoder
        hs_pad, hlens, _ = self.enc(xs_pad, ilens)
        lpz = None

        # 2. Decoder
        hlens = torch.tensor(list(map(int, hlens)))  # make sure hlens is tensor
        y = self.dec.recognize_beam_batch(
            hs_pad, hlens, lpz, trans_args, char_list, rnnlm
        )

        if prev:
            self.train()
        return y

    def calculate_all_attentions(self, xs_pad, ilens, ys_pad, ys_pad_src):
        """E2E attention calculation.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad:
            batch of padded character id sequence tensor (B, Lmax)
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
            att_ws = self.dec.calculate_all_attentions(
                hpad, hlens, ys_pad, lang_ids=tgt_lang_ids
            )

        return att_ws

    def subsample_frames(self, x):
        """Subsample speeh frames in the encoder."""
        # subsample frame
        x = x[:: self.subsample[0], :]
        ilen = [x.shape[0]]
        h = to_device(self, torch.from_numpy(np.array(x, dtype=np.float32)))
        h.contiguous()
        return h, ilen

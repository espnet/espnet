# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""RNN sequence-to-sequence speech translation model (pytorch)."""

import argparse
import copy
import logging
import math
import os
from itertools import groupby

import chainer
import nltk
import numpy as np
import six
import torch
from chainer import reporter

from espnet.nets.e2e_asr_common import label_smoothing_dist
from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.initialization import (
    lecun_normal_init_parameters, set_forget_bias_to_one)
from espnet.nets.pytorch_backend.nets_utils import (get_subsample, pad_list,
                                                    to_device, to_torch_tensor)
from espnet.nets.pytorch_backend.rnn.argument import \
    add_arguments_rnn_attention_common  # noqa: H301
from espnet.nets.pytorch_backend.rnn.argument import \
    add_arguments_rnn_decoder_common  # noqa: H301
from espnet.nets.pytorch_backend.rnn.argument import \
    add_arguments_rnn_encoder_common  # noqa: H301
from espnet.nets.pytorch_backend.rnn.attentions import att_for
from espnet.nets.pytorch_backend.rnn.decoders import decoder_for
from espnet.nets.pytorch_backend.rnn.encoders import encoder_for
from espnet.nets.st_interface import STInterface
from espnet.utils.fill_missing_args import fill_missing_args

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
        group = add_arguments_rnn_encoder_common(group)
        return parser

    @staticmethod
    def attention_add_arguments(parser):
        """Add arguments for the attention."""
        group = parser.add_argument_group("E2E attention setting")
        group = add_arguments_rnn_attention_common(group)
        return parser

    @staticmethod
    def decoder_add_arguments(parser):
        """Add arguments for the decoder."""
        group = parser.add_argument_group("E2E decoder setting")
        group = add_arguments_rnn_decoder_common(group)
        return parser

    def get_total_subsampling_factor(self):
        """Get total subsampling factor."""
        return self.enc.conv_subsampling_factor * int(np.prod(self.subsample))

    def __init__(self, idim, odim, args):
        """Construct an E2E object.

        :param int idim: dimension of inputs
        :param int odim: dimension of outputs
        :param Namespace args: argument Namespace containing options
        """
        super(E2E, self).__init__()
        torch.nn.Module.__init__(self)

        # fill missing arguments for compatibility
        args = fill_missing_args(args, self.add_arguments)

        self.asr_weight = args.asr_weight
        self.mt_weight = args.mt_weight
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
        # in ASR. However, blank labels are not used in MT.
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

        # multilingual related
        self.multilingual = getattr(args, "multilingual", False)
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
        :param torch.Tensor ys_pad: batch of padded token id sequence tensor (B, Lmax)
        :return: loss value
        :rtype: torch.Tensor
        """
        # 0. Extract target language ID
        if self.multilingual:
            tgt_lang_ids = ys_pad[:, 0:1]
            ys_pad = ys_pad[:, 1:]  # remove target language ID in the beginning
        else:
            tgt_lang_ids = None

        # 1. Encoder
        hs_pad, hlens, _ = self.enc(xs_pad, ilens)

        # 2. ST attention loss
        self.loss_st, self.acc, _ = self.dec(
            hs_pad, hlens, ys_pad, lang_ids=tgt_lang_ids
        )

        # 3. ASR loss
        (
            self.loss_asr_att,
            acc_asr,
            self.loss_asr_ctc,
            cer_ctc,
            cer,
            wer,
        ) = self.forward_asr(hs_pad, hlens, ys_pad_src)

        # 4. MT attention loss
        self.loss_mt, acc_mt = self.forward_mt(ys_pad, ys_pad_src)

        # 5. Compute BLEU
        if self.training or not self.report_bleu:
            self.bleu = 0.0
        else:
            lpz = None

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
            list_of_refs = []
            hyps = []
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

                hyps += [seq_hat_text.split(" ")]
                list_of_refs += [[seq_true_text.split(" ")]]

            self.bleu = nltk.bleu_score.corpus_bleu(list_of_refs, hyps) * 100

        asr_ctc_weight = self.mtlalpha
        self.loss_asr = (
            asr_ctc_weight * self.loss_asr_ctc
            + (1 - asr_ctc_weight) * self.loss_asr_att
        )
        self.loss = (
            (1 - self.asr_weight - self.mt_weight) * self.loss_st
            + self.asr_weight * self.loss_asr
            + self.mt_weight * self.loss_mt
        )
        loss_st_data = float(self.loss_st)
        loss_asr_data = float(self.loss_asr)
        loss_mt_data = float(self.loss_mt)
        loss_data = float(self.loss)
        if loss_data < CTC_LOSS_THRESHOLD and not math.isnan(loss_data):
            self.reporter.report(
                loss_asr_data,
                loss_mt_data,
                loss_st_data,
                acc_asr,
                acc_mt,
                self.acc,
                cer_ctc,
                cer,
                wer,
                self.bleu,
                loss_data,
            )
        else:
            logging.warning("loss (=%f) is not correct", loss_data)
        return self.loss

    def forward_asr(self, hs_pad, hlens, ys_pad):
        """Forward pass in the auxiliary ASR task.

        :param torch.Tensor hs_pad: batch of padded source sequences (B, Tmax, idim)
        :param torch.Tensor hlens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded target sequences (B, Lmax)
        :return: ASR attention loss value
        :rtype: torch.Tensor
        :return: accuracy in ASR attention decoder
        :rtype: float
        :return: ASR CTC loss value
        :rtype: torch.Tensor
        :return: character error rate from CTC prediction
        :rtype: float
        :return: character error rate from attetion decoder prediction
        :rtype: float
        :return: word error rate from attetion decoder prediction
        :rtype: float
        """
        import editdistance

        loss_att, loss_ctc = 0.0, 0.0
        acc = None
        cer, wer = None, None
        cer_ctc = None
        if self.asr_weight == 0:
            return loss_att, acc, loss_ctc, cer_ctc, cer, wer

        # attention
        if self.mtlalpha < 1:
            loss_asr, acc_asr, _ = self.dec_asr(hs_pad, hlens, ys_pad)

            # Compute wer and cer
            if not self.training and (self.report_cer or self.report_wer):
                if self.mtlalpha > 0 and self.recog_args.ctc_weight > 0.0:
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

                    seq_hat = [
                        self.char_list[int(idx)] for idx in y_hat if int(idx) != -1
                    ]
                    seq_true = [
                        self.char_list[int(idx)] for idx in y_true if int(idx) != -1
                    ]
                    seq_hat_text = "".join(seq_hat).replace(self.recog_args.space, " ")
                    seq_hat_text = seq_hat_text.replace(self.recog_args.blank, "")
                    seq_true_text = "".join(seq_true).replace(
                        self.recog_args.space, " "
                    )

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

        # CTC
        if self.mtlalpha > 0:
            loss_ctc = self.ctc(hs_pad, hlens, ys_pad)

            # Compute cer with CTC prediction
            if self.char_list is not None:
                cers = []
                y_hats = self.ctc.argmax(hs_pad).data
                for i, y in enumerate(y_hats):
                    y_hat = [x[0] for x in groupby(y)]
                    y_true = ys_pad[i]

                    seq_hat = [
                        self.char_list[int(idx)] for idx in y_hat if int(idx) != -1
                    ]
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

        return loss_att, acc, loss_ctc, cer_ctc, cer, wer

    def forward_mt(self, xs_pad, ys_pad):
        """Forward pass in the auxiliary MT task.

        :param torch.Tensor xs_pad: batch of padded source sequences (B, Tmax, idim)
        :param torch.Tensor ys_pad: batch of padded target sequences (B, Lmax)
        :return: MT loss value
        :rtype: torch.Tensor
        :return: accuracy in MT decoder
        :rtype: float
        """
        loss = 0.0
        acc = 0.0
        if self.mt_weight == 0:
            return loss, acc

        ilens = torch.sum(xs_pad != -1, dim=1).cpu().numpy()
        # NOTE: xs_pad is padded with -1
        ys_src = [y[y != -1] for y in xs_pad]  # parse padded ys_src
        xs_zero_pad = pad_list(ys_src, self.pad)  # re-pad with zero
        hs_pad, hlens, _ = self.enc_mt(
            self.dropout_mt(self.embed_mt(xs_zero_pad)), ilens
        )
        loss, acc, _ = self.dec(hs_pad, hlens, ys_pad)
        return loss, acc

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
        logging.info("input lengths: " + str(x.shape[0]))
        hs = self.encode(x).unsqueeze(0)
        logging.info("encoder output lengths: " + str(hs.size(1)))

        # 2. Decoder
        # decode the first utterance
        y = self.dec.recognize_beam(hs[0], None, trans_args, char_list, rnnlm)
        return y

    def translate_batch(self, xs, trans_args, char_list, rnnlm=None):
        """E2E batch beam search.

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

        # 2. Decoder
        hlens = torch.tensor(list(map(int, hlens)))  # make sure hlens is tensor
        y = self.dec.recognize_beam_batch(
            hs_pad, hlens, None, trans_args, char_list, rnnlm
        )

        if prev:
            self.train()
        return y

    def calculate_all_attentions(self, xs_pad, ilens, ys_pad, ys_pad_src):
        """E2E attention calculation.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded token id sequence tensor (B, Lmax)
        :param torch.Tensor ys_pad_src:
            batch of padded token id sequence tensor (B, Lmax)
        :return: attention weights with the following shape,
            1) multi-head case => attention weights (B, H, Lmax, Tmax),
            2) other case => attention weights (B, Lmax, Tmax).
        :rtype: float ndarray
        """
        self.eval()
        with torch.no_grad():
            # 1. Encoder
            if self.multilingual:
                tgt_lang_ids = ys_pad[:, 0:1]
                ys_pad = ys_pad[:, 1:]  # remove target language ID in the beginning
            else:
                tgt_lang_ids = None
            hpad, hlens, _ = self.enc(xs_pad, ilens)

            # 2. Decoder
            att_ws = self.dec.calculate_all_attentions(
                hpad, hlens, ys_pad, lang_ids=tgt_lang_ids
            )
        self.train()
        return att_ws

    def calculate_all_ctc_probs(self, xs_pad, ilens, ys_pad, ys_pad_src):
        """E2E CTC probability calculation.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded token id sequence tensor (B, Lmax)
        :param torch.Tensor
            ys_pad_src: batch of padded token id sequence tensor (B, Lmax)
        :return: CTC probability (B, Tmax, vocab)
        :rtype: float ndarray
        """
        probs = None
        if self.asr_weight == 0 or self.mtlalpha == 0:
            return probs

        self.eval()
        with torch.no_grad():
            # 1. Encoder
            hpad, hlens, _ = self.enc(xs_pad, ilens)

            # 2. CTC probs
            probs = self.ctc.softmax(hpad).cpu().numpy()
        self.train()
        return probs

    def subsample_frames(self, x):
        """Subsample speeh frames in the encoder."""
        # subsample frame
        x = x[:: self.subsample[0], :]
        ilen = [x.shape[0]]
        h = to_device(self, torch.from_numpy(np.array(x, dtype=np.float32)))
        h.contiguous()
        return h, ilen

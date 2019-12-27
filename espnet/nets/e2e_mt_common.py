#!/usr/bin/env python3

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Common functions for MT."""

import nltk
import numpy as np


class ErrorCalculator(object):
    """Calculate BLEU for E2E_ST and NMT models during training.

    :param y_hats: numpy array with predicted text
    :param y_pads: numpy array with true (target) text
    :param char_list: vocabulary list
    :param sym_space: space symbol
    :param sym_pad: pad symbol
    :param report_bleu: report BLUE score if True
    """

    def __init__(self, char_list, sym_space, sym_pad, report_bleu=False):
        """Construct an ErrorCalculator object."""
        super(ErrorCalculator, self).__init__()
        self.char_list = char_list
        self.space = sym_space
        self.pad = sym_pad
        self.report_bleu = report_bleu
        if self.space in self.char_list:
            self.idx_space = self.char_list.index(self.space)
        else:
            self.idx_space = None

    def __call__(self, ys_hat, ys_pad):
        """Calculate sentence-level BLEU score.

        :param torch.Tensor ys_hat: prediction (batch, seqlen)
        :param torch.Tensor ys_pad: reference (batch, seqlen)
        :return: sentence-level BLEU score
        :rtype float
        """
        bleu = None
        if not self.report_bleu:
            return bleu

        seqs_hat, seqs_true = self.convert_to_char(ys_hat, ys_pad)
        if self.report_bleu:
            bleu = self.calculate_bleu(seqs_hat, seqs_true)
        return bleu

    def convert_to_char(self, ys_hat, ys_pad):
        """Convert index to character.

        :param torch.Tensor seqs_hat: prediction (batch, seqlen)
        :param torch.Tensor seqs_true: reference (batch, seqlen)
        :return: token list of prediction
        :rtype list
        :return: token list of reference
        :rtype list
        """
        seqs_hat, seqs_true = [], []
        for i, y_hat in enumerate(ys_hat):
            y_true = ys_pad[i]
            eos_true = np.where(y_true == -1)[0]
            eos_true = eos_true[0] if len(eos_true) > 0 else len(y_true)
            # To avoid wrong lower BLEU than the one obtained from the decoding
            # eos from y_true is used to mark the eos in y_hat
            # because of that y_hats has not padded outs with -1.
            seq_hat = [self.char_list[int(idx)] for idx in y_hat[:eos_true]]
            seq_true = [self.char_list[int(idx)] for idx in y_true if int(idx) != -1]
            seq_hat_text = "".join(seq_hat).replace(self.space, ' ')
            seq_hat_text = seq_hat_text.replace(self.pad, '')
            seq_true_text = "".join(seq_true).replace(self.space, ' ')
            seqs_hat.append(seq_hat_text)
            seqs_true.append(seq_true_text)
        return seqs_hat, seqs_true

    def calculate_bleu(self, seqs_hat, seqs_true):
        """Calculate average sentence-level BLEU score.

        :param list seqs_hat: prediction
        :param list seqs_true: reference
        :return: average sentence-level BLEU score
        :rtype float
        """
        bleus = []
        for i, seq_hat_text in enumerate(seqs_hat):
            seq_true_text = seqs_true[i]
            bleu = nltk.bleu_score.sentence_bleu([seq_true_text], seq_hat_text) * 100
            bleus.append(bleu)
        return sum(bleus) / len(bleus)

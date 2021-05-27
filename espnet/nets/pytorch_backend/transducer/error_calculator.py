#!/usr/bin/env python3
# encoding: utf-8

"""CER/WER monitoring for transducer models."""

import editdistance

from espnet.nets.beam_search_transducer import BeamSearchTransducer


class ErrorCalculator(object):
    """Calculate CER and WER for transducer models.

    Args:
        decoder (torch.nn.Module|TransducerDecoderInterface): decoder module
        joint_network (torch.nn.Module): joint network module
        token_list (list): list of tokens
        sym_space (str): space symbol
        sym_blank (str): blank symbol
        report_cer (boolean): compute CER option
        report_wer (boolean): compute WER option

    """

    def __init__(
        self,
        decoder,
        joint_network,
        token_list,
        sym_space,
        sym_blank,
        report_cer=False,
        report_wer=False,
    ):
        """Construct an ErrorCalculator object for transducer model."""
        super().__init__()

        self.beam_search = BeamSearchTransducer(
            decoder=decoder,
            joint_network=joint_network,
            beam_size=1,
        )

        self.decoder = decoder

        self.token_list = token_list
        self.space = sym_space
        self.blank = sym_blank

        self.report_cer = report_cer
        self.report_wer = report_wer

    def __call__(self, hs_pad, ys_pad):
        """Calculate sentence-level WER/CER score for transducer models.

        Args:
            hs_pad (torch.Tensor): batch of padded input sequence (batch, T, D)
            ys_pad (torch.Tensor): reference (batch, seqlen)

        Returns:
            (float): sentence-level CER score
            (float): sentence-level WER score

        """
        cer, wer = None, None

        batchsize = int(hs_pad.size(0))
        batch_nbest = []

        hs_pad = hs_pad.to(next(self.decoder.parameters()).device)

        for b in range(batchsize):
            nbest_hyps = self.beam_search(hs_pad[b])
            batch_nbest.append(nbest_hyps[-1])

        ys_hat = [nbest_hyp.yseq[1:] for nbest_hyp in batch_nbest]

        seqs_hat, seqs_true = self.convert_to_char(ys_hat, ys_pad.cpu())

        if self.report_cer:
            cer = self.calculate_cer(seqs_hat, seqs_true)

        if self.report_wer:
            wer = self.calculate_wer(seqs_hat, seqs_true)

        return cer, wer

    def convert_to_char(self, ys_hat, ys_pad):
        """Convert index to character.

        Args:
            ys_hat (torch.Tensor): prediction (batch, seqlen)
            ys_pad (torch.Tensor): reference (batch, seqlen)

        Returns:
            (list): token list of prediction
            (list): token list of reference

        """
        seqs_hat, seqs_true = [], []

        for i, y_hat in enumerate(ys_hat):
            y_true = ys_pad[i]

            seq_hat = [self.token_list[int(idx)] for idx in y_hat]
            seq_true = [self.token_list[int(idx)] for idx in y_true if int(idx) != -1]

            seq_hat_text = "".join(seq_hat).replace(self.space, " ")
            seq_hat_text = seq_hat_text.replace(self.blank, "")
            seq_true_text = "".join(seq_true).replace(self.space, " ")

            seqs_hat.append(seq_hat_text)
            seqs_true.append(seq_true_text)

        return seqs_hat, seqs_true

    def calculate_cer(self, seqs_hat, seqs_true):
        """Calculate sentence-level CER score for transducer model.

        Args:
            seqs_hat (torch.Tensor): prediction (batch, seqlen)
            seqs_true (torch.Tensor): reference (batch, seqlen)

        Returns:
            (float): average sentence-level CER score

        """
        char_eds, char_ref_lens = [], []

        for i, seq_hat_text in enumerate(seqs_hat):
            seq_true_text = seqs_true[i]

            hyp_chars = seq_hat_text.replace(" ", "")
            ref_chars = seq_true_text.replace(" ", "")

            char_eds.append(editdistance.eval(hyp_chars, ref_chars))
            char_ref_lens.append(len(ref_chars))

        return float(sum(char_eds)) / sum(char_ref_lens)

    def calculate_wer(self, seqs_hat, seqs_true):
        """Calculate sentence-level WER score for transducer model.

        Args:
            seqs_hat (torch.Tensor): prediction (batch, seqlen)
            seqs_true (torch.Tensor): reference (batch, seqlen)

        Returns:
            (float): average sentence-level WER score

        """
        word_eds, word_ref_lens = [], []

        for i, seq_hat_text in enumerate(seqs_hat):
            seq_true_text = seqs_true[i]

            hyp_words = seq_hat_text.split()
            ref_words = seq_true_text.split()

            word_eds.append(editdistance.eval(hyp_words, ref_words))
            word_ref_lens.append(len(ref_words))

        return float(sum(word_eds)) / sum(word_ref_lens)

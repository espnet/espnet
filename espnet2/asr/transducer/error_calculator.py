"""Error Calculator module for Transducer."""

from typing import List, Tuple

import torch

from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.transducer.beam_search_transducer import BeamSearchTransducer


class ErrorCalculatorTransducer(object):
    """
        Error Calculator for Transducer-based models in speech recognition.

    This class calculates Character Error Rate (CER) and Word Error Rate (WER) for
    transducer models used in automatic speech recognition tasks.

    Attributes:
        beam_search (BeamSearchTransducer): Beam search decoder for transducer models.
        decoder (AbsDecoder): Decoder module.
        token_list (List[int]): List of token IDs.
        space (str): Space symbol.
        blank (str): Blank symbol.
        report_cer (bool): Flag to compute CER.
        report_wer (bool): Flag to compute WER.

    Args:
        decoder (AbsDecoder): Decoder module.
        joint_network (torch.nn.Module): Joint network module.
        token_list (List[int]): List of token IDs.
        sym_space (str): Space symbol.
        sym_blank (str): Blank symbol.
        report_cer (bool, optional): Whether to compute CER. Defaults to False.
        report_wer (bool, optional): Whether to compute WER. Defaults to False.

    Example:
        >>> decoder = TransducerDecoder(...)
        >>> joint_network = JointNetwork(...)
        >>> token_list = ["<blank>", "a", "b", "c", ...]
        >>> error_calc = ErrorCalculatorTransducer(
        ...     decoder, joint_network, token_list, sym_space="<space>", sym_blank="<blank>"
        ... )
        >>> cer, wer = error_calc(encoder_out, target)

    Note:
        This class uses beam search decoding to generate predictions and then
        calculates the error rates by comparing them with the target sequences.
    """

    def __init__(
        self,
        decoder: AbsDecoder,
        joint_network: torch.nn.Module,
        token_list: List[int],
        sym_space: str,
        sym_blank: str,
        report_cer: bool = False,
        report_wer: bool = False,
    ):
        """Construct an ErrorCalculatorTransducer."""
        super().__init__()

        self.beam_search = BeamSearchTransducer(
            decoder=decoder,
            joint_network=joint_network,
            beam_size=2,
            search_type="default",
            score_norm=False,
        )

        self.decoder = decoder

        self.token_list = token_list
        self.space = sym_space
        self.blank = sym_blank

        self.report_cer = report_cer
        self.report_wer = report_wer

    def __call__(self, encoder_out: torch.Tensor, target: torch.Tensor):
        """
                Calculate sentence-level WER/CER score for Transducer model.

        This method performs beam search decoding on the encoder output and calculates
        the Character Error Rate (CER) and Word Error Rate (WER) by comparing the
        decoded sequences with the target sequences.

        Args:
            encoder_out (torch.Tensor): Encoder output sequences. Shape: (B, T, D_enc),
                where B is the batch size, T is the sequence length, and D_enc is the
                encoder output dimension.
            target (torch.Tensor): Target label ID sequences. Shape: (B, L), where B is
                the batch size and L is the target sequence length.

        Returns:
            tuple: A tuple containing:
                - cer (float or None): Sentence-level Character Error Rate if report_cer
                  is True, else None.
                - wer (float or None): Sentence-level Word Error Rate if report_wer is
                  True, else None.

        Example:
            >>> encoder_out = torch.randn(2, 100, 256)  # Batch size 2, seq length 100
            >>> target = torch.randint(0, 1000, (2, 50))  # Batch size 2, target length 50
            >>> error_calc = ErrorCalculatorTransducer(...)
            >>> cer, wer = error_calc(encoder_out, target)
            >>> print(f"CER: {cer}, WER: {wer}")

        Note:
            This method uses the beam search algorithm to decode the encoder output.
            The resulting predictions are then converted to character sequences for
            error rate calculation.
        """
        cer, wer = None, None

        batchsize = int(encoder_out.size(0))
        batch_nbest = []

        encoder_out = encoder_out.to(next(self.decoder.parameters()).device)

        for b in range(batchsize):
            nbest_hyps = self.beam_search(encoder_out[b])
            batch_nbest.append(nbest_hyps)

        pred = [nbest_hyp[0].yseq[1:] for nbest_hyp in batch_nbest]

        char_pred, char_target = self.convert_to_char(pred, target)

        if self.report_cer:
            cer = self.calculate_cer(char_pred, char_target)

        if self.report_wer:
            wer = self.calculate_wer(char_pred, char_target)

        return cer, wer

    def convert_to_char(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[List, List]:
        """
                Convert label ID sequences to character sequences.

        This method transforms the predicted and target label ID sequences into
        character sequences by mapping each ID to its corresponding token. It also
        handles the replacement of space and blank symbols.

        Args:
            pred (torch.Tensor): Prediction label ID sequences. Shape: (B, U), where B
                is the batch size and U is the length of the predicted sequence.
            target (torch.Tensor): Target label ID sequences. Shape: (B, L), where B is
                the batch size and L is the length of the target sequence.

        Returns:
            tuple: A tuple containing two lists:
                - char_pred (List[str]): List of predicted character sequences, one for
                  each sample in the batch.
                - char_target (List[str]): List of target character sequences, one for
                  each sample in the batch.

        Example:
            >>> pred = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> target = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> error_calc = ErrorCalculatorTransducer(...)
            >>> char_pred, char_target = error_calc.convert_to_char(pred, target)
            >>> print(f"Predictions: {char_pred}")
            >>> print(f"Targets: {char_target}")

        Note:
            This method replaces the space symbol with an actual space character and
            removes any occurrence of the blank symbol in both predicted and target
            sequences.
        """
        char_pred, char_target = [], []

        for i, pred_i in enumerate(pred):
            char_pred_i = [self.token_list[int(h)] for h in pred_i]
            char_target_i = [self.token_list[int(r)] for r in target[i]]

            char_pred_i = "".join(char_pred_i).replace(self.space, " ")
            char_pred_i = char_pred_i.replace(self.blank, "")

            char_target_i = "".join(char_target_i).replace(self.space, " ")
            char_target_i = char_target_i.replace(self.blank, "")

            char_pred.append(char_pred_i)
            char_target.append(char_target_i)

        return char_pred, char_target

    def calculate_cer(
        self, char_pred: torch.Tensor, char_target: torch.Tensor
    ) -> float:
        """
                Calculate sentence-level Character Error Rate (CER) score.

        This method computes the CER by comparing the predicted character sequences
        with the target character sequences. It uses the Levenshtein distance
        (edit distance) to measure the difference between sequences.

        Args:
            char_pred (List[str]): List of predicted character sequences, one for
                each sample in the batch.
            char_target (List[str]): List of target character sequences, one for
                each sample in the batch.

        Returns:
            float: Average sentence-level CER score across the batch.

        Example:
            >>> char_pred = ["hello wrld", "how r u"]
            >>> char_target = ["hello world", "how are you"]
            >>> error_calc = ErrorCalculatorTransducer(...)
            >>> cer = error_calc.calculate_cer(char_pred, char_target)
            >>> print(f"Character Error Rate: {cer}")

        Note:
            - This method removes all space characters from both predicted and target
              sequences before calculating the edit distance.
            - The CER is calculated as the sum of edit distances divided by the sum of
              target sequence lengths.
            - This method requires the 'editdistance' package to be installed.
        """
        import editdistance

        distances, lens = [], []

        for i, char_pred_i in enumerate(char_pred):
            pred = char_pred_i.replace(" ", "")
            target = char_target[i].replace(" ", "")

            distances.append(editdistance.eval(pred, target))
            lens.append(len(target))

        return float(sum(distances)) / sum(lens)

    def calculate_wer(
        self, char_pred: torch.Tensor, char_target: torch.Tensor
    ) -> float:
        """
                Calculate sentence-level Word Error Rate (WER) score.

        This method computes the WER by comparing the predicted word sequences
        with the target word sequences. It uses the Levenshtein distance
        (edit distance) to measure the difference between word sequences.

        Args:
            char_pred (List[str]): List of predicted character sequences, one for
                each sample in the batch.
            char_target (List[str]): List of target character sequences, one for
                each sample in the batch.

        Returns:
            float: Average sentence-level WER score across the batch.

        Example:
            >>> char_pred = ["hello world how are you", "this is a test"]
            >>> char_target = ["hello world how are you doing", "this is only a test"]
            >>> error_calc = ErrorCalculatorTransducer(...)
            >>> wer = error_calc.calculate_wer(char_pred, char_target)
            >>> print(f"Word Error Rate: {wer}")

        Note:
            - This method splits each character sequence into words before calculating
              the edit distance.
            - The WER is calculated as the sum of edit distances divided by the sum of
              target word sequence lengths.
            - This method requires the 'editdistance' package to be installed.
        """
        import editdistance

        distances, lens = [], []

        for i, char_pred_i in enumerate(char_pred):
            pred = char_pred_i.split()
            target = char_target[i].split()

            distances.append(editdistance.eval(pred, target))
            lens.append(len(target))

        return float(sum(distances)) / sum(lens)

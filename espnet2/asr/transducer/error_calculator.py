"""Error Calculator module for Transducer."""

from typing import List, Tuple

import torch

from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.transducer.beam_search_transducer import BeamSearchTransducer


class ErrorCalculatorTransducer(object):
    """
    Error Calculator module for Transducer.

    This class calculates Character Error Rate (CER) and Word Error Rate (WER)
    for transducer models. It utilizes a decoder module and a joint network
    to perform beam search for predictions, and then compares those predictions
    against target sequences to compute error rates.

    Attributes:
        decoder (AbsDecoder): The decoder module used for generating predictions.
        token_list (List[int]): List of tokens used for converting IDs to characters.
        sym_space (str): The symbol representing space in the token list.
        sym_blank (str): The symbol representing blank in the token list.
        report_cer (bool): Flag indicating whether to compute CER.
        report_wer (bool): Flag indicating whether to compute WER.

    Args:
        decoder: An instance of AbsDecoder, the decoder module.
        joint_network: A torch.nn.Module representing the joint network.
        token_list: A list of integer tokens corresponding to characters.
        sym_space: A string representing the space symbol.
        sym_blank: A string representing the blank symbol.
        report_cer: A boolean indicating whether to report CER (default: False).
        report_wer: A boolean indicating whether to report WER (default: False).

    Examples:
        >>> error_calculator = ErrorCalculatorTransducer(
        ...     decoder=my_decoder,
        ...     joint_network=my_joint_network,
        ...     token_list=my_token_list,
        ...     sym_space='<space>',
        ...     sym_blank='<blank>',
        ...     report_cer=True,
        ...     report_wer=True
        ... )
        >>> cer, wer = error_calculator(encoder_output, target_labels)
        >>> print(f"CER: {cer}, WER: {wer}")

    Raises:
        ValueError: If the input shapes of encoder_out and target are not compatible.

    Note:
        The predictions and targets are processed as character sequences, where
        blank symbols are ignored and space symbols are converted to spaces.

    Todo:
        - Implement logging for better debugging and tracking of error calculations.
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
        """Calculate sentence-level WER/CER score for Transducer model.

        Args:
            encoder_out: Encoder output sequences. (B, T, D_enc)
            target: Target label ID sequences. (B, L)

        Returns:
            : Sentence-level CER score.
            : Sentence-level WER score.

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

        This method takes the predicted and target label ID sequences and
        converts them into their corresponding character representations
        based on the provided token list. It handles special symbols,
        replacing space symbols with spaces and removing blank symbols.

        Args:
            pred: Prediction label ID sequences. Shape: (B, U), where B is the
                batch size and U is the maximum length of predicted sequences.
            target: Target label ID sequences. Shape: (B, L), where L is the
                maximum length of target sequences.

        Returns:
            Tuple[List[str], List[str]]:
                char_pred: List of prediction character sequences. Shape: (B, ?).
                char_target: List of target character sequences. Shape: (B, ?).

        Examples:
            >>> token_list = ['a', 'b', ' ', '_']
            >>> pred = torch.tensor([[0, 1, 2], [1, 0, 3]])
            >>> target = torch.tensor([[0, 1], [1, 2]])
            >>> ec_transducer = ErrorCalculatorTransducer(decoder, joint_network,
            ...                                             token_list, ' ', '_')
            >>> char_pred, char_target = ec_transducer.convert_to_char(pred, target)
            >>> print(char_pred)
            ['ab ', 'ba']
            >>> print(char_target)
            ['ab', 'a ']

        Note:
            The output character sequences may vary in length based on the
            predictions and targets. The '?' in the shape notation indicates
            that the length may differ for each sequence.
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

        This method computes the CER by comparing predicted character
        sequences against target character sequences. The CER is defined
        as the edit distance between the predicted and target sequences
        divided by the length of the target sequence, ignoring spaces.

        Args:
            char_pred: Prediction character sequences. Shape: (B, ?), where B
                is the batch size and ? is the variable length of the
                predicted sequences.
            char_target: Target character sequences. Shape: (B, ?), where B
                is the batch size and ? is the variable length of the
                target sequences.

        Returns:
            float: Average sentence-level CER score across the batch.

        Raises:
            ZeroDivisionError: If the total length of the target sequences
                is zero.

        Examples:
            >>> char_pred = ["hello", "world"]
            >>> char_target = ["hallo", "word"]
            >>> cer = calculate_cer(char_pred, char_target)
            >>> print(cer)
            0.25  # Example output based on edit distance calculation

        Note:
            The method uses the `editdistance` library to compute the edit
            distance between character sequences. Make sure to have it
            installed in your environment.
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
        Calculate sentence-level WER score.

        This method computes the Word Error Rate (WER) based on the predicted
        character sequences and the target character sequences. WER is defined
        as the number of word-level errors divided by the total number of words
        in the reference (target) text. The errors can be substitutions,
        deletions, or insertions.

        Args:
            char_pred: Prediction character sequences. (B, ?)
            char_target: Target character sequences. (B, ?)

        Returns:
            float: Average sentence-level WER score.

        Examples:
            >>> char_pred = ["this is a test", "hello world"]
            >>> char_target = ["this is test", "hello there world"]
            >>> wer_score = calculate_wer(char_pred, char_target)
            >>> print(wer_score)
            0.25

        Note:
            The WER calculation uses the edit distance algorithm to evaluate
            the differences between predicted and target sequences. It is
            recommended to preprocess the input sequences to ensure consistent
            formatting (e.g., removing extra spaces).
        """
        import editdistance

        distances, lens = [], []

        for i, char_pred_i in enumerate(char_pred):
            pred = char_pred_i.split()
            target = char_target[i].split()

            distances.append(editdistance.eval(pred, target))
            lens.append(len(target))

        return float(sum(distances)) / sum(lens)

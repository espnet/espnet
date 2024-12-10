"""Error Calculator module for Transducer."""

from typing import List, Optional, Tuple

import torch

from espnet2.asr_transducer.beam_search_transducer import BeamSearchTransducer
from espnet2.asr_transducer.decoder.abs_decoder import AbsDecoder
from espnet2.asr_transducer.joint_network import JointNetwork


class ErrorCalculator:
    """
    Error Calculator module for Transducer.

    This module provides the `ErrorCalculator` class which is responsible for 
    calculating the Character Error Rate (CER) and Word Error Rate (WER) for 
    transducer models in Automatic Speech Recognition (ASR).

    Attributes:
        decoder (AbsDecoder): The decoder module used for generating predictions.
        joint_network (JointNetwork): The joint network module used in the model.
        token_list (List[int]): List of token units for mapping predictions to 
            characters.
        space (str): Symbol representing space in the token list.
        blank (str): Symbol representing the blank in the token list.
        report_cer (bool): Flag indicating whether to compute CER.
        report_wer (bool): Flag indicating whether to compute WER.
        
    Args:
        decoder (AbsDecoder): Decoder module.
        joint_network (JointNetwork): Joint Network module.
        token_list (List[int]): List of token units.
        sym_space (str): Space symbol.
        sym_blank (str): Blank symbol.
        nstep (int, optional): Maximum number of symbol expansions at each time step 
            with mAES. Defaults to 2.
        report_cer (bool, optional): Whether to compute CER. Defaults to False.
        report_wer (bool, optional): Whether to compute WER. Defaults to False.

    Examples:
        # Initialize the ErrorCalculator
        error_calculator = ErrorCalculator(
            decoder=decoder_instance,
            joint_network=joint_network_instance,
            token_list=[0, 1, 2, 3],  # Example token list
            sym_space=' ',
            sym_blank='|',
            nstep=2,
            report_cer=True,
            report_wer=True
        )

        # Calculate CER and WER
        cer, wer = error_calculator(encoder_out, target, encoder_out_lens)

    Note:
        The `ErrorCalculator` uses the mAES algorithm for validation to ensure 
        better performance and control over the number of emitted symbols during 
        validation.

    Raises:
        ValueError: If the lengths of the predictions and targets do not match.
    """

    def __init__(
        self,
        decoder: AbsDecoder,
        joint_network: JointNetwork,
        token_list: List[int],
        sym_space: str,
        sym_blank: str,
        nstep: int = 2,
        report_cer: bool = False,
        report_wer: bool = False,
    ) -> None:
        """Construct an ErrorCalculatorTransducer object."""
        super().__init__()

        # (b-flo): Since the commit #8c9c851 we rely on the mAES algorithm for
        # validation instead of the default algorithm.
        #
        # With the addition of k2 pruned transducer loss, the number of emitted symbols
        # at each timestep can be restricted during training. Performing an unrestricted
        # (/ unconstrained) decoding without regard to the training conditions can lead
        # to huge performance degradation. It won't be an issue with mAES and the user
        # can now control the number of emitted symbols during validation.
        #
        # Also, under certain conditions, using the default algorithm can lead to a long
        # decoding procedure due to the loop break condition. Other algorithms,
        # such as mAES, won't be impacted by that.
        self.beam_search = BeamSearchTransducer(
            decoder=decoder,
            joint_network=joint_network,
            beam_size=2,
            search_type="maes",
            nstep=nstep,
            score_norm=False,
        )

        self.decoder = decoder

        self.token_list = token_list
        self.space = sym_space
        self.blank = sym_blank

        self.report_cer = report_cer
        self.report_wer = report_wer

    def __call__(
        self,
        encoder_out: torch.Tensor,
        target: torch.Tensor,
        encoder_out_lens: torch.Tensor,
    ) -> Tuple[Optional[float], Optional[float]]:
        """Calculate sentence-level WER or/and CER score for Transducer model.

        Args:
            encoder_out: Encoder output sequences. (B, T, D_enc)
            target: Target label ID sequences. (B, L)
            encoder_out_lens: Encoder output sequences length. (B,)

        Returns:
            : Sentence-level CER score.
            : Sentence-level WER score.

        """
        cer, wer = None, None

        batchsize = int(encoder_out.size(0))

        encoder_out = encoder_out.to(next(self.decoder.parameters()).device)

        batch_nbest = [
            self.beam_search(encoder_out[b][: encoder_out_lens[b]])
            for b in range(batchsize)
        ]
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

        This method takes in prediction and target label ID sequences and converts 
        them into their corresponding character sequences. It replaces space and 
        blank symbols as specified by the user.

        Args:
            pred: Prediction label ID sequences. Shape: (B, U), where B is the batch 
                size and U is the number of predicted symbols.
            target: Target label ID sequences. Shape: (B, L), where L is the number 
                    of target symbols.

        Returns:
            Tuple[List, List]: 
                - char_pred: Prediction character sequences. Shape: (B, ?), where ? 
                is the variable length of character sequences.
                - char_target: Target character sequences. Shape: (B, ?).

        Examples:
            >>> pred = torch.tensor([[1, 2, 3], [4, 5, 0]])
            >>> target = torch.tensor([[1, 3], [4, 6]])
            >>> token_list = ['<blank>', 'a', 'b', 'c', 'd', 'e', 'f', ' ']
            >>> calculator = ErrorCalculator(decoder, joint_network, token_list, 
            ...                              ' ', '<blank>')
            >>> char_pred, char_target = calculator.convert_to_char(pred, target)
            >>> print(char_pred)   # Output: ['abc', 'de']
            >>> print(char_target)  # Output: ['ac', 'd']

        Note:
            - The space symbol is replaced with a regular space character (' ').
            - The blank symbol is removed from the output character sequences.
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
        Error Calculator module for Transducer.

        This module provides functionality to calculate Character Error Rate (CER) 
        and Word Error Rate (WER) for transducer models. It includes methods to 
        process output from the model and compute the error rates based on 
        predicted and target sequences.

        Attributes:
            decoder: An instance of the decoder module used in the transducer model.
            joint_network: An instance of the joint network module.
            token_list: A list of token units for conversion between IDs and characters.
            space: A symbol representing space in the token list.
            blank: A symbol representing blank in the token list.
            report_cer: A flag indicating whether to compute CER.
            report_wer: A flag indicating whether to compute WER.

        Args:
            decoder: An instance of AbsDecoder, the decoder module.
            joint_network: An instance of JointNetwork, the joint network module.
            token_list: A list of integer token units.
            sym_space: A string representing the space symbol.
            sym_blank: A string representing the blank symbol.
            nstep: An integer specifying the maximum number of symbol expansions 
                at each time step with mAES (default is 2).
            report_cer: A boolean indicating whether to compute CER (default is False).
            report_wer: A boolean indicating whether to compute WER (default is False).

        Examples:
            error_calculator = ErrorCalculator(decoder, joint_network, token_list, 
                                            sym_space, sym_blank, report_cer=True, 
                                            report_wer=True)
            cer_score, wer_score = error_calculator(encoder_out, target, encoder_out_lens)

        Raises:
            ValueError: If the input tensors do not have the expected dimensions.

        Note:
            The CER and WER calculations are based on the edit distance between 
            predicted and target sequences, which is computed using the editdistance 
            library.

        Todo:
            - Optimize performance for large batch sizes.
            - Implement logging for error calculation results.
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
        Error Calculator module for Transducer.

        This module provides functionality to calculate Character Error Rate (CER) 
        and Word Error Rate (WER) for transducer models. It utilizes a beam search 
        decoder to generate predictions based on the encoder outputs and compares 
        these predictions with target sequences to compute the respective error rates.

        Attributes:
            decoder: An instance of AbsDecoder used for decoding.
            joint_network: An instance of JointNetwork used in the transducer model.
            token_list: A list of token units representing characters or words.
            sym_space: The symbol representing space in the token list.
            sym_blank: The symbol representing blank in the token list.
            nstep: The maximum number of symbol expansions at each time step for mAES.
            report_cer: A boolean indicating whether to compute CER.
            report_wer: A boolean indicating whether to compute WER.

        Args:
            decoder: Decoder module.
            joint_network: Joint Network module.
            token_list: List of token units.
            sym_space: Space symbol.
            sym_blank: Blank symbol.
            nstep: Maximum number of symbol expansions at each time step w/ mAES.
            report_cer: Whether to compute CER.
            report_wer: Whether to compute WER.

        Examples:
            >>> error_calculator = ErrorCalculator(decoder, joint_network, token_list,
            ...                                     ' ', '<blank>', nstep=2,
            ...                                     report_cer=True, report_wer=True)
            >>> cer, wer = error_calculator(encoder_out, target, encoder_out_lens)
            >>> print(f"CER: {cer}, WER: {wer}")

        Note:
            The calculation relies on the mAES algorithm for validation instead of 
            the default algorithm to avoid performance degradation during training.

        Raises:
            ValueError: If the dimensions of encoder_out or target are inconsistent.
        """
        import editdistance

        distances, lens = [], []

        for i, char_pred_i in enumerate(char_pred):
            pred = char_pred_i.replace("▁", " ").split()
            target = char_target[i].replace("▁", " ").split()

            distances.append(editdistance.eval(pred, target))
            lens.append(len(target))

        return float(sum(distances)) / sum(lens)

from typing import Any, Dict, Optional, Tuple, Union

import torch
from typeguard import typechecked

from espnet2.tts2.feats_extract.abs_feats_extract import AbsFeatsExtractDiscrete


class IdentityFeatureExtract(AbsFeatsExtractDiscrete):
    """
    IdentityFeatureExtract is a feature extraction class that keeps the input
    discrete sequence unchanged. It is designed for use in text-to-speech
    (TTS) systems within the ESPnet framework. This class inherits from
    AbsFeatsExtractDiscrete and overrides the forward method to validate and
    return the input data.

    Attributes:
        None

    Args:
        None

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - A tensor containing the input converted to long type.
            - A tensor containing the input lengths.

    Raises:
        AssertionError: If the input tensor is complex, floating point, or
        boolean, or if it does not have 2 dimensions, or if the number of
        input sequences does not match the number of input lengths.

    Examples:
        >>> extractor = IdentityFeatureExtract()
        >>> input_tensor = torch.tensor([[1, 2], [3, 4]])
        >>> input_lengths = torch.tensor([2, 2])
        >>> output, lengths = extractor.forward(input_tensor, input_lengths)
        >>> print(output)
        tensor([[1, 2],
                [3, 4]])
        >>> print(lengths)
        tensor([2, 2])

    Note:
        This class is primarily intended for use where the input sequence
        needs to be passed through without modification.
    """

    @typechecked
    def __init__(self):
        super().__init__()

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[Any, Dict]:
        """
            Forward pass of the IdentityFeatureExtract class.

        This method processes the input tensor and returns it along with its lengths.
        It ensures that the input tensor meets certain criteria, specifically that it
        is a 2-dimensional tensor of integer type.

        Args:
            input (torch.Tensor): A 2D tensor containing the discrete input sequence.
            input_lengths (torch.Tensor): A 1D tensor containing the lengths of the
                input sequences. Its size must match the first dimension of the input.

        Returns:
            Tuple[Any, Dict]: A tuple containing:
                - The input tensor converted to long type.
                - The input lengths tensor.

        Raises:
            AssertionError: If the input tensor is complex, floating point, or boolean,
                or if the input tensor does not have 2 dimensions, or if the number of
                input sequences does not match the number of lengths.

        Examples:
            >>> extractor = IdentityFeatureExtract()
            >>> input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> input_lengths = torch.tensor([3, 3])
            >>> output, lengths = extractor.forward(input_tensor, input_lengths)
            >>> print(output)
            tensor([[1, 2, 3],
                    [4, 5, 6]])
            >>> print(lengths)
            tensor([3, 3])

        Note:
            This class is designed to keep the input discrete sequence unchanged.
        """
        # torch doesn't have .is_int() function
        assert (
            not input.is_complex()
            and not input.is_floating_point()
            and not input.dtype == torch.bool
        ), "Invalid data type."
        assert input.dim() == 2, "Input should have 2 dimensions."
        assert input.size(0) == input_lengths.size(0), "Invalid lengths."

        return input.long(), input_lengths

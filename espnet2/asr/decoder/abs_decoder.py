from abc import ABC, abstractmethod
from typing import Tuple

import torch

from espnet.nets.scorer_interface import ScorerInterface


class AbsDecoder(torch.nn.Module, ScorerInterface, ABC):
    """
        Abstract base class for decoders in speech recognition models.

    This class defines the interface for decoders used in the ESPnet framework.
    It inherits from torch.nn.Module for neural network functionality,
    ScorerInterface for scoring methods, and ABC for abstract base class behavior.

    Attributes:
        None

    Note:
        Subclasses must implement the `forward` method.

    Examples:
        >>> class MyDecoder(AbsDecoder):
        ...     def forward(self, hs_pad, hlens, ys_in_pad, ys_in_lens):
        ...         # Implement decoder logic here
        ...         pass
        ...
        ...     def score(self, ys, state, x):
        ...         # Implement scoring logic here
        ...         pass
    """

    @abstractmethod
    def forward(
        self,
        hs_pad: torch.Tensor,
        hlens: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
                Forward pass of the decoder.

        This abstract method defines the forward pass for the decoder. It takes
        encoded features and target sequences as input and produces decoded output.

        Args:
            hs_pad (torch.Tensor): Padded hidden state sequences from the encoder.
                Shape: (batch, time, hidden_dim)
            hlens (torch.Tensor): Lengths of hidden state sequences.
                Shape: (batch,)
            ys_in_pad (torch.Tensor): Padded input token sequences for teacher forcing.
                Shape: (batch, output_length)
            ys_in_lens (torch.Tensor): Lengths of input token sequences.
                Shape: (batch,)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Decoded output sequences. Shape: (batch, output_length, vocab_size)
                - Attention weights or None. Shape: (batch, output_length, input_length)

        Raises:
            NotImplementedError: This method must be implemented by subclasses.

        Note:
            This method should be implemented by all subclasses of AbsDecoder.

        Examples:
            >>> class MyDecoder(AbsDecoder):
            ...     def forward(self, hs_pad, hlens, ys_in_pad, ys_in_lens):
            ...         # Decoder implementation
            ...         decoded_output = ...  # Shape: (batch, output_length, vocab_size)
            ...         attention_weights = ...  # Shape: (batch, output_length, input_length)
            ...         return decoded_output, attention_weights
        """
        raise NotImplementedError

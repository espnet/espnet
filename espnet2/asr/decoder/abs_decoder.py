from abc import ABC, abstractmethod
from typing import Tuple

import torch

from espnet.nets.scorer_interface import ScorerInterface


class AbsDecoder(torch.nn.Module, ScorerInterface, ABC):
    """
    Abstract base class for ASR (Automatic Speech Recognition) decoders in the ESPnet2
    framework. This class defines the interface for decoders that process the hidden
    states of an encoder and generate output sequences. It inherits from
    `torch.nn.Module` and `ScorerInterface`, providing a foundation for various
    decoder implementations.

    Attributes:
        None

    Args:
        hs_pad (torch.Tensor): A tensor of shape (batch_size, max_time, num_units)
            containing the padded hidden states from the encoder.
        hlens (torch.Tensor): A tensor of shape (batch_size,) representing the lengths
            of the hidden state sequences (before padding).
        ys_in_pad (torch.Tensor): A tensor of shape (batch_size, max_target_length)
            containing the padded input target sequences (e.g., ground truth labels).
        ys_in_lens (torch.Tensor): A tensor of shape (batch_size,) representing the
            lengths of the input target sequences (before padding).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - logits (torch.Tensor): A tensor of shape (batch_size, max_target_length,
            num_classes) representing the output logits for each target token.
            - attentions (torch.Tensor): A tensor of shape (batch_size, max_target_length,
            max_time) representing the attention weights.

    Raises:
        NotImplementedError: If the forward method is called directly on an instance
        of AbsDecoder without being overridden by a subclass.

    Examples:
        >>> decoder = MyDecoder()  # MyDecoder should be a concrete implementation of AbsDecoder
        >>> hs_pad = torch.randn(32, 100, 256)  # Example hidden states
        >>> hlens = torch.randint(1, 100, (32,))
        >>> ys_in_pad = torch.randint(0, 10, (32, 50))  # Example target sequences
        >>> ys_in_lens = torch.randint(1, 50, (32,))
        >>> logits, attentions = decoder(hs_pad, hlens, ys_in_pad, ys_in_lens)

    Note:
        This class should not be instantiated directly. It is intended to be subclassed
        by specific decoder implementations that provide concrete behavior in the
        `forward` method.
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
        Executes the forward pass of the AbsDecoder.

        This method takes encoded sequences and target sequences as input and
        computes the output of the decoder. It is an abstract method that must
        be implemented by subclasses of AbsDecoder. The method processes the
        input tensors, typically representing hidden states and target sequences,
        and returns the output tensors along with any relevant state information.

        Args:
            hs_pad (torch.Tensor): A padded tensor of shape (B, T, D) containing
                the hidden states from the encoder, where B is the batch size,
                T is the maximum sequence length, and D is the dimensionality
                of the hidden states.
            hlens (torch.Tensor): A tensor of shape (B,) representing the actual
                lengths of the encoder outputs for each sequence in the batch.
            ys_in_pad (torch.Tensor): A padded tensor of shape (B, S) containing
                the input sequences to the decoder, where S is the maximum
                target sequence length.
            ys_in_lens (torch.Tensor): A tensor of shape (B,) representing the
                actual lengths of the input sequences for the decoder.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - output (torch.Tensor): A tensor of shape (B, S, V) representing
                the decoder output probabilities over the vocabulary V.
                - state (torch.Tensor): A tensor representing the internal state
                of the decoder after processing the input.

        Raises:
            NotImplementedError: If the method is called directly from the
            AbsDecoder class without being overridden in a subclass.

        Examples:
            # Example of how to use the forward method in a subclass
            class MyDecoder(AbsDecoder):
                def forward(self, hs_pad, hlens, ys_in_pad, ys_in_lens):
                    # Implement the forward pass logic here
                    pass

            decoder = MyDecoder()
            hs_pad = torch.randn(10, 5, 256)  # Example hidden states
            hlens = torch.tensor([5] * 10)    # Example lengths
            ys_in_pad = torch.randint(0, 100, (10, 7))  # Example input
            ys_in_lens = torch.tensor([7] * 10)          # Example lengths
            output, state = decoder.forward(hs_pad, hlens, ys_in_pad, ys_in_lens)
        """
        raise NotImplementedError

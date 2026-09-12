"""Abstract interface for speech-tokenizer quantizers."""

from abc import ABC, abstractmethod

import torch

from espnet2.tok.output import SpeechTokenizerOutput


class AbsSpeechTokenizerQuantizer(torch.nn.Module, ABC):
    """Define the common contract for differentiable speech quantizers.

    Implementations are expected to provide differentiable assignments for
    training, hard integer token IDs for inference, and the corresponding
    sequence lengths.
    """

    @property
    @abstractmethod
    def num_clusters(self) -> int:
        """Return the number of discrete clusters."""
        raise NotImplementedError

    @abstractmethod
    def forward(
        self, features: torch.Tensor, feature_lengths: torch.Tensor
    ) -> SpeechTokenizerOutput:
        """Quantize continuous features with a differentiable assignment."""
        raise NotImplementedError

    @abstractmethod
    def encode(
        self, features: torch.Tensor, feature_lengths: torch.Tensor
    ) -> SpeechTokenizerOutput:
        """Quantize continuous features deterministically."""
        raise NotImplementedError

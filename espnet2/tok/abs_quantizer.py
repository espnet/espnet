"""Abstract interface for speech-tokenizer quantizers."""

from abc import ABC, abstractmethod

import torch

from espnet2.tok.output import SpeechTokenizerOutput


class AbsSpeechTokenizerQuantizer(torch.nn.Module, ABC):
    """Define the common contract for differentiable speech quantizers.

    Implementations are expected to provide differentiable assignments for
    training, hard integer token IDs for inference, and the corresponding
    sequence lengths. This interface discretizes continuous features while
    preserving a gradient path for training. The similarly named
    ``speechlm.tokenizer.AbsTokenizer`` is intended for no-grad postprocessing
    of generated tokens, such as codec codes to waveform or BPE tokens to text.
    Use that interface for SpeechLM postprocessing and this one when
    discretization must preserve gradients.
    """

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Return the expected dimension of input features."""
        raise NotImplementedError

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

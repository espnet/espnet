"""Abstract interface for speech reconstruction objectives."""

from abc import abstractmethod
from typing import Dict, Tuple

import torch

from espnet2.tok.objective.abs_objective import (
    AbsSpeechTokenizerObjective,
)
from espnet2.tok.output import SpeechTokenizerOutput


class AbsReconstructionObjective(AbsSpeechTokenizerObjective):
    """Define reconstruction learning and synthesis from shared speech tokens.

    Reconstruction is a downstream tokenizer objective with an additional
    synthesis interface.  Adversarial implementations may also expose a
    discriminator loss that is excluded from the main optimizer.
    """

    @property
    def is_adversarial(self) -> bool:
        """Return whether this objective requires a discriminator optimizer."""
        return False

    def forward_discriminator(
        self,
        tokenizer_output: SpeechTokenizerOutput,
        **batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Compute discriminator loss for an adversarial objective."""
        raise NotImplementedError(
            f"{type(self).__name__} is not an adversarial reconstruction objective"
        )

    @abstractmethod
    def synthesize(
        self,
        tokenizer_output: SpeechTokenizerOutput,
        **batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Synthesize outputs and return them with their valid lengths."""
        raise NotImplementedError

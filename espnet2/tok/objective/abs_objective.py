"""Abstract interface for speech-tokenizer objectives."""

from abc import ABC, abstractmethod
from typing import Dict, Tuple

import torch

from espnet2.tok.output import SpeechTokenizerOutput


class AbsSpeechTokenizerObjective(torch.nn.Module, ABC):
    """Define an auxiliary learning objective over shared speech tokens.

    Objectives consume a common tokenizer output and contribute a weighted loss
    to the main optimizer.  New downstream tasks can implement this boundary
    without changing the shared tokenizer or top-level loss aggregation.
    """

    @abstractmethod
    def forward(
        self,
        tokenizer_output: SpeechTokenizerOutput,
        **batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Compute an objective from shared token representations."""
        raise NotImplementedError

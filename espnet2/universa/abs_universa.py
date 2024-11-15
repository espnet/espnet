# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Universa abstrast class."""

from abc import ABC, abstractmethod
from typing import Dict, Tuple

import torch


class AbsUniversa(torch.nn.Module, ABC):
    """Universa abstract class."""

    @abstractmethod
    def forward(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor,
        metrics: torch.Tensor,
        ref_audio: torch.Tensor,
        ref_audio_lengths: torch.Tensor,
        ref_text: torch.Tensor,
        ref_text_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Calculate outputs and return the loss tensor."""
        raise NotImplementedError

    @abstractmethod
    def inference(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Return predicted output as a dict."""
        raise NotImplementedError

    @property
    def require_raw_audio(self):
        """Return whether or not raw_audio is required."""
        return False

    @property
    def require_raw_text(self):
        """Return whether or not raw_text is required."""
        return False
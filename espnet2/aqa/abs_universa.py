# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Universa abstract class."""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch


class AbsUniversa(torch.nn.Module, ABC):
    """Universa abstract class."""

    use_ref_audio: bool = False
    use_ref_text: bool = False

    @abstractmethod
    def forward(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor,
        metrics: Dict[str, torch.Tensor],
        ref_audio: Optional[torch.Tensor] = None,
        ref_audio_lengths: Optional[torch.Tensor] = None,
        ref_text: Optional[torch.Tensor] = None,
        ref_text_lengths: Optional[torch.Tensor] = None,
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
    ) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
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

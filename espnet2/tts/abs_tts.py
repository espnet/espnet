from abc import ABC
from abc import abstractmethod
from typing import Dict
from typing import Tuple

import torch


class AbsTTS(torch.nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        spembs: torch.Tensor = None,
        spcs: torch.Tensor = None,
        spcs_lengths: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def inference(
        self,
        text: torch.Tensor,
        spembs: torch.Tensor = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @property
    def require_raw_speech(self):
        """Return whether or not raw_speech is required."""
        return False

    @property
    def require_vocoder(self):
        """Return whether or not vocoder is required."""
        return True

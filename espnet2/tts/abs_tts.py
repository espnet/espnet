from abc import ABC
from abc import abstractmethod
from typing import Dict
from typing import Tuple
from typing import Union

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
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    @property
    def require_vocoder(self):
        """Return whether or not vocoder is required."""
        return True

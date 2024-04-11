from abc import ABC, abstractmethod
from typing import Tuple, Dict

import torch


class AbsPredictor(torch.nn.Module, ABC):
    """The abstract Speech LM class"""

    @abstractmethod
    def forward(
        self, input: torch.Tensor, input_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        raise NotImplementedError

    @abstractmethod
    def inference(
        self, prefix: torch.Tensor, input_mask: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        raise NotImplementedError

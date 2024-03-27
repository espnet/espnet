from abc import ABC, abstractmethod
from typing import Tuple

import torch


class AbsMasker(torch.nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

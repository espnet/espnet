from abc import ABC, abstractmethod
from typing import Tuple

import torch


class AbsLoss(torch.nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        xs_pad: torch.Tensor,
        ys_pad: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

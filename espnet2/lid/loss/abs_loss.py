from abc import abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn as nn


class AbsLoss(nn.Module):
    def __init__(self, nout: int, **kwargs):
        super().__init__()

    @abstractmethod
    def forward(
        self, input: torch.Tensor, label: Optional[torch.Tensor] = None
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        raise NotImplementedError

from abc import ABC, abstractmethod
from typing import Tuple

import torch


class AbsDecoder(torch.nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        hs_pad: torch.Tensor,
        hlens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

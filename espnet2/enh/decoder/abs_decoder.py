from abc import ABC, abstractmethod
from typing import Tuple

import torch


class AbsDecoder(torch.nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        input: torch.Tensor,
        ilens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @property
    @abstractmethod
    def frame_size(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def hop_size(self) -> int:
        raise NotImplementedError

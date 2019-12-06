from abc import ABC, abstractmethod

import torch
from typing import Tuple


class AbsFrontend(torch.nn.Module, ABC):
    @abstractmethod
    def out_dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

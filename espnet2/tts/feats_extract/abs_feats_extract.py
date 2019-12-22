from abc import ABC, abstractmethod

import torch
from typing import Tuple


class AbsFeatsExtract(torch.nn.Module, ABC):
    @abstractmethod
    def output_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def build_griffin_lim_vocoder(self, griffin_lim_iters):
        raise NotImplementedError

    @abstractmethod
    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class AbsPooling(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def output_size(self) -> int:
        raise NotImplementedError

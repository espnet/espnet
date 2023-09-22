from abc import ABC, abstractmethod

import torch


class AbsPooling(torch.nn.Module, ABC):
    @abstractmethod
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

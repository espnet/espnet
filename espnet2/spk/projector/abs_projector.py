from abc import ABC, abstractmethod

import torch


class AbsProjector(torch.nn.Module, ABC):
    @abstractmethod
    def output_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def forward(self, utt_embd: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

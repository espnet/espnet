from abc import ABC, abstractmethod

import torch


class AbsDiffusion(torch.nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        input: torch.Tensor,
        ilens: torch.Tensor,
    ):
        raise NotImplementedError

    @abstractmethod
    def enhance(self, input: torch.Tensor):
        raise NotImplementedError

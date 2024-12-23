from abc import ABC, abstractmethod

import torch


class AbsTTS(torch.nn.Module, ABC):
    @abstractmethod
    def warmup(self):
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        text: str,
    ):
        raise NotImplementedError

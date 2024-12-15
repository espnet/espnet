from abc import ABC, abstractmethod

import torch


class AbsVAD(torch.nn.Module, ABC):
    @abstractmethod
    def warmup(self):
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        speech,
        sample_rate,
    ):
        raise NotImplementedError

from abc import ABC
from abc import abstractmethod
from collections import OrderedDict
from typing import Tuple

import torch


class AbsSeparator(torch.nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        input: torch.Tensor,
        ilens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        raise NotImplementedError

    @property
    @abstractmethod
    def input_dim(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def bottleneck_dim(self) -> int:
        raise NotImplementedError
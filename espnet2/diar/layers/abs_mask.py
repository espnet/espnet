from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Tuple

import torch


class AbsMask(torch.nn.Module, ABC):
    @property
    @abstractmethod
    def max_num_spk(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        input,
        ilens,
        bottleneck_feat,
        num_spk,
    ) -> Tuple[Tuple[torch.Tensor], torch.Tensor, OrderedDict]:
        raise NotImplementedError

# Copyright 2021 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from abc import ABC
from abc import abstractmethod
from collections import OrderedDict
from typing import Tuple

import torch


class AbsDiarization(torch.nn.Module, ABC):
    # @abstractmethod
    # def output_size(self) -> int:
    #     raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        input: torch.Tensor,
        ilens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, OrderedDict]:
        raise NotImplementedError

    @abstractmethod
    def forward_rawwav(
        self, input: torch.Tensor, ilens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, OrderedDict]:
        raise NotImplementedError

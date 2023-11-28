from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Dict, Optional, Tuple

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

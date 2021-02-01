from abc import ABC
from abc import abstractmethod
from typing import Tuple

import torch

from espnet.nets.scorer_interface import ScorerInterface


class AbsDecoder(torch.nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        input: torch.Tensor,
        ilens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
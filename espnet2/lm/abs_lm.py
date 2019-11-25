from abc import abstractmethod, ABC
from typing import Tuple

import torch

from espnet.nets.scorer_interface import ScorerInterface


class AbsLM(torch.nn.Module, ScorerInterface, ABC):
    @abstractmethod
    def forward(self, input: torch.Tensor, hidden: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

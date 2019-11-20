from abc import abstractmethod, ABC
from typing import Tuple

import torch

from espnet.nets.scorer_interface import ScorerInterface


class AbsLM(ABC, ScorerInterface, torch.nn.Module):
    @abstractmethod
    def forward(self, input: torch.Tensor, hidden: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

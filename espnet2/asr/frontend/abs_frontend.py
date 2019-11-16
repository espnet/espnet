from abc import ABC, abstractmethod

import torch
from typing import Tuple


class AbsFrontend(ABC, torch.nn.Module):
    @abstractmethod
    def forward(self, input: torch.Tensor, input_lengths: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

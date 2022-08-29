from abc import ABC, abstractmethod
from typing import Tuple

import torch


class AbsAttractor(torch.nn.Module, ABC):
    @abstractmethod
    def forward(
        self, enc_input: torch.Tensor, ilens: torch.Tensor, dec_input: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

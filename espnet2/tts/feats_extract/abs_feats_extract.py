from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import torch


class AbsFeatsExtract(torch.nn.Module, ABC):
    @abstractmethod
    def output_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

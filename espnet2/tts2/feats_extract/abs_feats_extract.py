from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import torch


class AbsFeatsExtractDiscrete(torch.nn.Module, ABC):
    """Parse the discrete token sequence

    into structured data format for predicting. E.g.,
    (1) keep as sequence
    (2) resize as a matrix
    (3) multi-resolution ...
    """

    @abstractmethod
    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[Any, Dict]:
        raise NotImplementedError

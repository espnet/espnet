from abc import ABC
from abc import abstractmethod
from typing import Dict
from typing import List
from typing import Tuple

import torch


class AbsLossWrapper(torch.nn.Module, ABC):
    """Base class for all Enhancement loss wrapper modules."""

    # The weight for the current loss in the multi-task learning.
    # The overall training target will be combined as:
    # loss = weight_1 * loss_1 + ... + weight_N * loss_N
    weight = 1.0

    @abstractmethod
    def forward(
        self,
        ref: List,
        inf: List,
        others: Dict,
    ) -> Tuple[torch.Tensor, Dict, Dict]:
        raise NotImplementedError

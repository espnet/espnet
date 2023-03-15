from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Tuple

import torch


class AbsExtractor(torch.nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        input: torch.Tensor,
        ilens: torch.Tensor,
        input_aux: torch.Tensor,
        ilens_aux: torch.Tensor,
        suffix_tag: str = "",
    ) -> Tuple[Tuple[torch.Tensor], torch.Tensor, OrderedDict]:
        raise NotImplementedError

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch


class AbsSegmenter(torch.nn.Module, ABC):
    @abstractmethod
    def pre_segment(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def logit_segment(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

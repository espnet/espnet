from abc import ABC
from abc import abstractmethod


import torch

EPS = torch.finfo(torch.get_default_dtype()).eps


class AbsEnhLoss(torch.nn.Module, ABC):
    """Base class for all Enhancement loss modules."""

    # the name will be the key that appears in the reporter
    @property
    def name(self) -> str:
        return NotImplementedError

    @abstractmethod
    def forward(
        self,
        ref,
        inf,
    ) -> torch.Tensor:
        # the return tensor should be shape of (batch)
        raise NotImplementedError

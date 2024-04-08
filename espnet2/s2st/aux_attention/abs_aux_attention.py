from abc import ABC, abstractmethod

import torch

EPS = torch.finfo(torch.get_default_dtype()).eps


class AbsS2STAuxAttention(torch.nn.Module, ABC):
    """Base class for all S2ST auxiliary attention modules.

    Refer to https://arxiv.org/abs/2107.08661
    """

    # the name will be the key that appears in the reporter
    @property
    def name(self) -> str:
        return NotImplementedError

    @abstractmethod
    def forward(
        self,
    ) -> torch.Tensor:
        # the return tensor should be shape of (batch)
        raise NotImplementedError

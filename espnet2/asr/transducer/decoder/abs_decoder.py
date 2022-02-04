from abc import ABC
from abc import abstractmethod

import torch


class AbsDecoder(torch.nn.Module, ABC):
    """Abstract decoder module."""

    @abstractmethod
    def set_device(self, device: torch.Tensor):
        """Set GPU device to use.

        Args:
            device: Device ID.

        """
        raise NotImplementedError

    @abstractmethod
    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        """Encode source label sequences.

        Args:
            labels: Label ID sequences. (B, L)

        Returns:
            dec_out: Decoder output sequences. (B, T, D_dec)

        """
        raise NotImplementedError

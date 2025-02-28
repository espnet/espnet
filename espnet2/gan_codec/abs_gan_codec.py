# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""GAN-based Neural Codec abstrast class."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Union

import torch


class AbsGANCodec(ABC, torch.nn.Module):
    """GAN-based Neural Codec model abstract class."""

    @abstractmethod
    def meta_info(self) -> Dict[str, Any]:
        """Return meta information of the codec."""
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        forward_generator,
        *args,
        **kwargs,
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor], int]]:
        """Return generator or discriminator loss."""
        raise NotImplementedError

    @abstractmethod
    def encode(
        self,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Return encoded codecs from waveform."""
        raise NotImplementedError

    @abstractmethod
    def decode(
        self,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Return decoded waveform from codecs."""
        raise NotImplementedError

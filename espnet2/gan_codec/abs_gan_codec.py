# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""GAN-based Neural Codec abstrast class."""

from abc import ABC, abstractmethod
from typing import Dict, Union

import torch


class AbsGANCodec(ABC, torch.nn.Module):
    """GAN-based Neural Codec model abstract class."""

    @abstractmethod
    def forward(
        self,
        forward_generator,
        *args,
        **kwargs,
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor], int]]:
        """Return generator or discriminator loss."""
        raise NotImplementedError

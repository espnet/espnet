# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""GAN-based SVS abstrast class."""

from abc import ABC, abstractmethod
from typing import Dict, Union

import torch

from espnet2.svs.abs_svs import AbsSVS


class AbsGANSVS(AbsSVS, ABC):
    """GAN-based SVS model abstract class."""

    @abstractmethod
    def forward(
        self,
        forward_generator,
        *args,
        **kwargs,
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor], int]]:
        """Return generator or discriminator loss."""
        raise NotImplementedError

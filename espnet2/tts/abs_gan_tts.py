# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""GAN-based TTS abstrast class."""

from abc import ABC
from abc import abstractmethod

from typing import Dict
from typing import Optional
from typing import Tuple

import torch


class AbsGANTTS(torch.nn.Module, ABC):
    @abstractmethod
    def forward_generator(
        self,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def forward_discrminator(
        self,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def inference(
        self,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        raise NotImplementedError

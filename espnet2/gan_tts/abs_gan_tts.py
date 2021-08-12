# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""GAN-based TTS abstrast class."""

from abc import ABC
from abc import abstractmethod

from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import torch


class AbsGANTTS(torch.nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        forward_generator,
        *args,
        **kwargs,
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor], int]]:
        raise NotImplementedError

    @abstractmethod
    def inference(
        self,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        raise NotImplementedError

    @property
    def require_raw_speech(self):
        """Return whether or not raw_speech is required."""
        return False

# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""GAN-based TTS abstrast class."""

from abc import ABC
from abc import abstractmethod

from typing import Dict
from typing import Union

import torch

from espnet2.tts.abs_tts import AbsTTS


class AbsGANTTS(AbsTTS, ABC):
    @abstractmethod
    def forward(
        self,
        forward_generator,
        *args,
        **kwargs,
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor], int]]:
        # NOTE(kan-bayashi): Require additional arguments forward_generator
        #   to switch the loss calculation of generator and discriminator.
        raise NotImplementedError

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Dict, Optional, Tuple

from espnet2.enh.diffusion.abs_diffusion import AbsDiffusion

import torch

from espnet2.train.class_choices import ClassChoices
from espnet2.enh.layers.dcunet import DCUNet

score_choices = ClassChoices(
    name="score_dnn",
    classes=dict(dcunet=DCUNet),
    type_check=torch.nn.Module,
    default="dcunet",
)


class ScoreModel(AbsDiffusion):

    def __init__(self, dnn, sde):

        self.dnn = dnn
        self.sde = sde

    @abstractmethod
    def forward(
        self,
        input: torch.Tensor,
        ilens: torch.Tensor,
    ):
        raise NotImplementedError

    @abstractmethod
    def enhance(self, input: torch.Tensor):
        raise NotImplementedError

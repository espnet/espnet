from collections import OrderedDict
from distutils.version import LooseVersion
from typing import List
from typing import Tuple
from typing import Union

import torch
from torch_complex.tensor import ComplexTensor
import numpy as np

from espnet2.enh.layers.complex_utils import is_complex
from espnet2.enh.layers.diffuse import DiffuSE
from espnet2.enh.layers.dprnn import merge_feature
from espnet2.enh.layers.dprnn import split_feature
from espnet2.enh.separator.abs_separator import AbsSeparator


is_torch_1_9_plus = LooseVersion(torch.__version__) >= LooseVersion("1.9.0")


class CDiffuSESeparator(AbsSeparator):
    def __init__(
        self,
        num_spk: int = 1,
        n_fft: int = 1024,
        residual_layers: int = 30,
        residual_channels: int = 64,
        dilation_cycle_length: int = 10,
        noise_schedule_length: int = 50,
    ):
        """Separator

        Args:
            num_spk: speaker number,
        """
        super().__init__()

        self._num_spk = num_spk
        self.diffuse = DiffuSE(
            n_fft=n_fft,
            residual_layers=residual_layers,
            residual_channels=residual_channels,
            dilation_cycle_length=dilation_cycle_length,
            noise_schedule_length=noise_schedule_length,
        )

    def forward(
        self,
        input: Union[torch.Tensor, ComplexTensor],
        conditioner: Union[torch.Tensor, ComplexTensor],
        diffusion_step: int,
        ilens: torch.Tensor,
    ) -> Tuple[List[Union[torch.Tensor, ComplexTensor]], torch.Tensor, OrderedDict]:
        """
        Forward.

        Args:

        Returns:
        """

        if is_complex(conditioner):
            feature = abs(conditioner)
        else:
            feature = conditioner

        processed = self.diffuse(input, conditioner, diffusion_step)

        processed = torch.transpose(processed, 0, 1)

        return processed, ilens, processed

    @property
    def num_spk(self):
        return self._num_spk

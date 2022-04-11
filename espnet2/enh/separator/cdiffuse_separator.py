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
        noise_schedule: list = np.linspace(1e-4, 0.035, 50).tolist(),
    ):
        """ Separator

        Args:
        """
        super().__init__()

        self._num_spk = num_spk
        self.diffuse = DiffuSE(
            n_fft=n_fft,
            residual_layers=residual_layers,
            residual_channels=residual_channels,
            dilation_cycle_length=dilation_cycle_length,
            noise_schedule=noise_schedule,
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
        # import pdb
        # pdb.set_trace()
        
        if is_complex(conditioner):
            feature = abs(conditioner)
        else:
            feature = conditioner
            
        processed = self.diffuse(input, conditioner, diffusion_step)
        
        processed = torch.transpose(processed,0,1)

        return processed, ilens, processed
        
        # # if complex spectrum,
        # if is_complex(input):
        #     feature = abs(input)
        # else:
        #     feature = input

        # B, T, N = feature.shape

        # feature = feature.transpose(1, 2)  # B, N, T
        # segmented, rest = split_feature(
        #     feature, segment_size=self.segment_size
        # )  # B, N, L, K

        # processed = self.dprnn(segmented)  # B, N*num_spk, L, K

        # processed = merge_feature(processed, rest)  # B, N*num_spk, T

        # processed = processed.transpose(1, 2)  # B, T, N*num_spk
        # processed = processed.view(B, T, N, self.num_spk)
        # masks = self.nonlinear(processed).unbind(dim=3)

        # masked = [input * m for m in masks]

        # others = OrderedDict(
        #     zip(["mask_spk{}".format(i + 1) for i in range(len(masks))], masks)
        # )

        # return masked, ilens, others

    @property
    def num_spk(self):
        return self._num_spk

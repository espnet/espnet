# Copyright 2023 Jee-weon Jung
# Apache 2.0

"""RawNet3 Encoder"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from typeguard import check_argument_types
from asteroid_filterbanks import Encoder, ParamSincFB

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.spk.layers.RawNetBasicBlock import *



class RawNet3Encoder(AbsEncoder):
    """
    RawNet3 encoder. Extracts frame-level RawNet embeddings from raw waveform.

    Args:
    """
    def __init__(
        self,
        output_size: int = 1024,
        block,
        model_scale,
        sinc_stride: int = 16,
        **kwargs,
    ):
        assesrt check_argument_types()
        super().__init__()
        self._output_size = output_size

        self.waveform_process = nn.Sequential(
            PreEmphasis(), nn.InstanceNorm1d(1, eps=1e-4, affine=True)
        )
        self.conv = Encoder(
            ParamSincFB(
                output_size // 4,
                251,
                stride=sinc_stride
            )
        )
        self.relu = nn.ReLU()

        self.layer1 = block(
            output_size // 4, C, kernel_size=3, dilation=, scale=model_scale, pool=5
        )
        self.layer2 = block(
            output_size, output_size, kernel_size=3, dilation=3, scale=model_scale, pool=3
        )
        self.layer3 = block(output_size, output_size, kernel_size=3, dilation=4, scale=model_scale)
        self.layer4 = nn.Conv1d(3 * output_size, 1536, kernel_size=1)

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        data: torch.Tensor,
    ):
        # waveform transformation and normalization here
        with torch.cuda.amp.autocast(enabled=False):
            x = self.waveform_process(data)
            x = torch.abs(self.conv(x))
            x = torch.log(x + 1e-6)
            x = x - torch.mean(x, dim=-1, keepdim=True)

        # frame-level propagation
        x1 = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(self.mp3(x1) + x2)

        x = self.layer4(torch.cat((self.mp3(x1), x2, x3), dim=1))
        x = self.relu(x)

        return x





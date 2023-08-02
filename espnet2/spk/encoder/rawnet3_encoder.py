# Copyright 2023 Jee-weon Jung
# Apache 2.0

"""RawNet3 Encoder"""

import torch
import torch.nn as nn
from typeguard import check_argument_types

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.spk.layers.rawnet_block import Bottle2neck


class RawNet3Encoder(AbsEncoder):
    """
    RawNet3 encoder. Extracts frame-level RawNet embeddings from raw waveform.
    paper: J. Jung et al., "Pushing the limits of raw waveform speaker
        recognition", in Proc. INTERSPEECH, 2022.

    Args:
        input_size: input feature dimension.
        block: type of encoder block class to use.
        model_scale: scale value of the Res2Net architecture.
        ndim: dimensionality of the hidden representation.
        output_size: ouptut embedding dimension.
    """

    def __init__(
        self,
        input_size: int,
        block: str = "Bottle2neck",
        model_scale: int = 8,
        ndim: int = 1024,
        output_size: int = 1536,
        **kwargs,
    ):
        assert check_argument_types()
        super().__init__()
        if block == "Bottle2neck":
            block = Bottle2neck
        else:
            raise ValueError(f"unsupported block, got: {block}")

        self._output_size = output_size

        self.relu = nn.ReLU()

        self.layer1 = block(
            input_size,
            ndim,
            kernel_size=3,
            dilation=2,
            scale=model_scale,
            pool=5,
        )
        self.layer2 = block(
            ndim,
            ndim,
            kernel_size=3,
            dilation=3,
            scale=model_scale,
            pool=3,
        )
        self.layer3 = block(ndim, ndim, kernel_size=3, dilation=4, scale=model_scale)
        self.layer4 = nn.Conv1d(3 * ndim, output_size, kernel_size=1)

        self.mp3 = nn.MaxPool1d(3)

    def output_size(self) -> int:
        return self._output_size

    def forward(self, x: torch.Tensor):
        # frame-level propagation
        x1 = self.layer1(x.permute(0, 2, 1))
        x2 = self.layer2(x1)
        x3 = self.layer3(self.mp3(x1) + x2)

        x = self.layer4(torch.cat((self.mp3(x1), x2, x3), dim=1))
        x = self.relu(x)

        return x

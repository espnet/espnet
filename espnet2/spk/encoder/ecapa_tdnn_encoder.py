# Copyright 2023 Jee-weon Jung
# Apache 2.0

"""
ECAPA-TDNN Encoder
"""

import torch
import torch.nn as nn
from typeguard import check_argument_types

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.spk.layers.ecapa_block import EcapaBlock


class EcapaTdnnEncoder(AbsEncoder):
    """
    ECAPA-TDNN encoder. Extracts frame-level ECAPA-TDNN embeddings from
    mel-filterbank energy or MFCC features.
    Paper: B Desplanques at el., ``ECAPA-TDNN: Emphasized Channel Attention,
        Propagation and Aggregation in TDNN Based Speaker Verification,''
        in Proc. INTERSPEECH, 2020.

    Args:
        input_size: input feature dimension.
        block: type of encoder block class to use.
        model_scale: scale value of the Res2Net architecture.
        ndim: dimensionality of the hidden representation.
        output_size: output embedding dimension.
    """

    def __init__(
        self,
        input_size: int,
        block: str = "EcapaBlock",
        model_scale: int = 8,
        ndim: int = 1024,
        output_size: int = 1536,
        **kwargs,
    ):
        assert check_argument_types()
        super().__init__()
        if block == "EcapaBlock":
            block = EcapaBlock
        else:
            raise ValueError(f"unsupported block, got: {block}")
        self._output_size = output_size

        self.conv = nn.Conv1d(input_size, ndim, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(ndim)

        self.layer1 = block(ndim, ndim, kernel_size=3, dilation=2, scale=model_scale)
        self.layer2 = block(ndim, ndim, kernel_size=3, dilation=3, scale=model_scale)
        self.layer3 = block(ndim, ndim, kernel_size=3, dilation=4, scale=model_scale)
        self.layer4 = nn.Conv1d(3 * ndim, output_size, kernel_size=1)

        self.mp3 = nn.MaxPool1d(3)

    def output_size(self) -> int:
        return self._output_size

    def forward(self, x: torch.Tensor):
        x = self.conv(x.permute(0, 2, 1))
        x = self.relu(x)
        x = self.bn(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x + x1)
        x3 = self.layer3(x + x1 + x2)

        x = self.layer4(torch.cat((x1, x2, x3), dim=1))
        x = self.relu(x)

        return x

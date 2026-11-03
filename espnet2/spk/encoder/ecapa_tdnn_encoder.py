# Copyright 2023 Jee-weon Jung
# Apache 2.0

"""
ECAPA-TDNN Encoder
"""

from typing import Optional, Sequence

import torch
import torch.nn as nn
from typeguard import typechecked

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.spk.layers.ecapa_block import EcapaBlock


class EcapaTdnnEncoder(AbsEncoder):
    """ECAPA-TDNN encoder. Extracts frame-level ECAPA-TDNN embeddings from

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
        num_blocks: number of SE-Res2 blocks, with dilations 2, 3, ...,
            num_blocks + 1. The default is the paper's three blocks (2, 3, 4).
            Four blocks (2, 3, 4, 5) follow the IDLab VoxSRC-20 system, which
            adds a fourth SE-Res2Block with dilation 5 (J. Thienpondt et al.,
            ``The IDLAB VoxSRC-20 Submission: Large Margin Fine-Tuning and
            Quality-Aware Score Calibration in DNN Based Speaker
            Verification,'' arXiv:2010.11255, Sec. 2). For five or more,
            choosing the dilations with ``dilations`` is recommended.
        dilations: dilation of each SE-Res2 block, such as ``[2, 3, 4, 2, 3, 4]``;
            the number of blocks is its length. Dilations do not change the
            parameter shapes, so build the model from the configuration it
            was trained with (the ``encoder_conf`` saved in its
            ``config.yaml``).

    Every block takes the first convolution's output plus the outputs of all
    earlier blocks, and the 1x1 convolution ``layer4`` aggregates all block
    outputs. The first three blocks keep the names ``layer1``-``layer3`` and
    further blocks are ``extra_layers``, so three-block checkpoints load
    unchanged.
    """

    @typechecked
    def __init__(
        self,
        input_size: int,
        block: str = "EcapaBlock",
        model_scale: int = 8,
        ndim: int = 1024,
        output_size: int = 1536,
        num_blocks: Optional[int] = None,
        dilations: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        if block == "EcapaBlock":
            block: type = EcapaBlock
        else:
            raise ValueError(f"unsupported block, got: {block}")
        self._output_size = output_size

        self.conv = nn.Conv1d(input_size, ndim, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(ndim)

        if dilations is None:
            dilations = range(2, (3 if num_blocks is None else num_blocks) + 2)
        elif num_blocks is not None and num_blocks != len(dilations):
            raise ValueError(
                f"num_blocks ({num_blocks}) differs from len(dilations) "
                f"({len(dilations)})"
            )
        dilations = list(dilations)
        if len(dilations) < 3:
            raise ValueError(f"at least 3 blocks are required, got: {dilations}")

        d1, d2, d3 = dilations[:3]
        self.layer1 = block(ndim, ndim, kernel_size=3, dilation=d1, scale=model_scale)
        self.layer2 = block(ndim, ndim, kernel_size=3, dilation=d2, scale=model_scale)
        self.layer3 = block(ndim, ndim, kernel_size=3, dilation=d3, scale=model_scale)
        self.extra_layers = nn.ModuleList(
            block(ndim, ndim, kernel_size=3, dilation=d, scale=model_scale)
            for d in dilations[3:]
        )
        self.layer4 = nn.Conv1d(len(dilations) * ndim, output_size, kernel_size=1)

        self.mp3 = nn.MaxPool1d(3)

    def output_size(self) -> int:
        return self._output_size

    def forward(self, x: torch.Tensor):
        """Calculate forward propagation.

        Args:
            x (torch.Tensor): Input tensor (#batch, L, input_size).

        Returns:
            torch.Tensor: Output tensor (#batch, L, output_size).

        """
        x = self.conv(x.permute(0, 2, 1))
        x = self.relu(x)
        x = self.bn(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x + x1)
        x3 = self.layer3(x + x1 + x2)
        outputs = [x1, x2, x3]
        if self.extra_layers:
            total = x + x1 + x2 + x3
            for layer in self.extra_layers:
                outputs.append(layer(total))
                total = total + outputs[-1]

        x = self.layer4(torch.cat(outputs, dim=1))
        x = self.relu(x)

        return x

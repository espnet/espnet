from typing import Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from typeguard import typechecked

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.spk.layers.resnet_block import BasicBlock, Bottleneck


class ResNetEncoder(AbsEncoder):
    """ResNet Encoder. Extracts frame-level ResNet embeddings from

    mel-filterbank energy or MFCC features.
    Paper: K. He et al., "Deep Residual Learning for Image Recognition",
    Adapted from https://github.com/wenet-e2e/wespeaker/blob/master/we
    -speaker/models/resnet.py

    Args:
        input_size: input feature dimension.
        block: type of encoder block class, either BasicBlock or Bottleneck.
        num_blocks: number of blocks in each layer.
        m_channels: number of channels in the first convolution layer.
    """

    @typechecked
    def __init__(
        self,
        input_size: int,
        block: Type[nn.Module] = BasicBlock,
        num_blocks: tuple = (2, 2, 2, 2),
        m_channels: int = 32,
        resnet_type: Optional[str] = None,
    ):
        super(ResNetEncoder, self).__init__()

        resnet_configs = {
            "resnet18": (BasicBlock, (2, 2, 2, 2)),
            "resnet34": (BasicBlock, (3, 4, 6, 3)),
            "resnet50": (Bottleneck, (3, 4, 6, 3)),
            "resnet101": (Bottleneck, (3, 4, 23, 3)),
            "resnet152": (Bottleneck, (3, 8, 36, 3)),
            "resnet221": (Bottleneck, (6, 16, 48, 3)),
            "resnet293": (Bottleneck, (10, 20, 64, 3)),
        }

        if resnet_type is not None:
            if resnet_type not in resnet_configs:
                raise ValueError(f"Unsupported resnet_type: {resnet_type}")
            block, num_blocks = resnet_configs[resnet_type]

        self.in_planes = m_channels
        self.stats_dim = int(input_size / 8) * m_channels * 8
        self._output_size = self.stats_dim * block.expansion

        self.conv1 = nn.Conv2d(
            1, m_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(m_channels)
        self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, m_channels * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, m_channels * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, m_channels * 8, num_blocks[3], stride=2)

    def _make_layer(
        self, block: Type[nn.Module], planes: int, num_blocks: int, stride: int
    ) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def output_size(self) -> int:
        return self._output_size

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = x.permute(0, 2, 1)  # (B,T,F) => (B,F,T)
        x = x.unsqueeze_(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = out.transpose(1, 3)
        out = torch.flatten(out, 2, -1)
        out = out.transpose(1, 2)  # (B, T, D)

        return out

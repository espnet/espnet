# Copyright 2021 Tomoki Hayashi
# Copyright 2022 Yifeng Yu
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Duration predictor modules in VISinger.
"""

import torch

from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm


class DurationPredictor(torch.nn.Module):
    def __init__(
        self, channels, filter_channels, kernel_size, dropout_rate, gin_channels=0
    ):
        super().__init__()

        self.in_channels = channels + 1
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.gin_channels = gin_channels

        self.drop = torch.nn.Dropout(dropout_rate)
        self.conv_1 = torch.nn.Conv1d(
            channels + 1, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_1 = LayerNorm(filter_channels, dim=1)
        self.conv_2 = torch.nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_2 = LayerNorm(filter_channels, dim=1)
        self.proj = torch.nn.Conv1d(filter_channels, 1, 1)

        if gin_channels != 0:
            self.cond = torch.nn.Conv1d(gin_channels, channels, 1)

    def forward(self, x, x_mask, beat_lab, g=None):
        x = torch.detach(x)
        beat_lab = torch.detach(beat_lab)
        beat_lab = beat_lab.unsqueeze(1)
        x = torch.cat((x, beat_lab), 1)

        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask

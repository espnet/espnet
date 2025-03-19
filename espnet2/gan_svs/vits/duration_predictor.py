# Copyright 2021 Tomoki Hayashi
# Copyright 2022 Yifeng Yu
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Duration predictor modules in VISinger."""

import torch

from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm


class DurationPredictor(torch.nn.Module):
    def __init__(
        self,
        channels,
        filter_channels,
        kernel_size,
        dropout_rate,
        global_channels=0,
    ):
        """Initialize duration predictor module.

        Args:
            channels (int): Number of input channels.
            filter_channels (int): Number of filter channels.
            kernel_size (int): Size of the convolutional kernel.
            dropout_rate (float): Dropout rate.
            global_channels (int, optional): Number of global conditioning channels.

        """
        super().__init__()

        self.in_channels = channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate

        self.drop = torch.nn.Dropout(dropout_rate)
        self.conv_1 = torch.nn.Conv1d(
            channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_1 = LayerNorm(filter_channels, dim=1)
        self.conv_2 = torch.nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_2 = LayerNorm(filter_channels, dim=1)
        self.conv_3 = torch.nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_3 = LayerNorm(filter_channels, dim=1)
        self.proj = torch.nn.Conv1d(filter_channels, 2, 1)

        if global_channels > 0:
            self.conv = torch.nn.Conv1d(global_channels, channels, 1)

    def forward(self, x, x_mask, g=None):
        """Forward pass through the duration predictor module.

        Args:
            x (Tensor): Input tensor (B, in_channels, T).
            x_mask (Tensor): Mask tensor (B, 1, T).
            g (Tensor, optional): Global condition tensor (B, global_channels, 1).

        Returns:
            Tensor: Predicted duration tensor (B, 2, T).
        """

        # multi-singer
        if g is not None:
            g = torch.detach(g)
            x = x + self.conv(g)

        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)

        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)

        x = self.conv_3(x * x_mask)
        x = torch.relu(x)
        x = self.norm_3(x)
        x = self.drop(x)

        x = self.proj(x * x_mask)
        return x * x_mask

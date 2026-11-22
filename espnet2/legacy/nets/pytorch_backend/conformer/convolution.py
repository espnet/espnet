#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Johns Hopkins University (Shinji Watanabe)
#                Northwestern Polytechnical University (Pengcheng Guo)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""ConvolutionModule definition."""

from torch import nn


def mask_padded_frames(x, mask_pad):
    """Zero the padded time steps of `x` (#batch, channels, time).

    `mask_pad` is (#batch, 1, time) with True/1 on real frames, or None. A mask
    whose time axis does not match `x` (the single-frame query of a cached
    step, for instance) is ignored, since it cannot be applied.
    """
    if (
        mask_pad is None
        or mask_pad.dim() != 3
        or mask_pad.size(1) != 1
        or mask_pad.size(-1) != x.size(-1)
    ):
        return x
    return x.masked_fill(mask_pad.eq(0), 0.0)


class ConvolutionModule(nn.Module):
    """ConvolutionModule in Conformer model.

    Args:
        channels (int): The number of channels of conv layers.
        kernel_size (int): Kernerl size of conv layers.

    """

    def __init__(self, channels, kernel_size, activation=nn.ReLU(), bias=True):
        """Construct an ConvolutionModule object."""
        super(ConvolutionModule, self).__init__()
        # kernerl_size should be a odd number for 'SAME' padding
        assert (kernel_size - 1) % 2 == 0

        self.pointwise_conv1 = nn.Conv1d(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=channels,
            bias=bias,
        )
        self.norm = nn.BatchNorm1d(channels)
        self.pointwise_conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.activation = activation

    def forward(self, x, mask_pad=None):
        """Compute convolution module.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, channels).
            mask_pad (torch.Tensor): Non-padding mask (#batch, 1, time), True
                or 1 where a frame is real. When given, the padded frames are
                zeroed right before the depthwise convolution, so the last
                real frames of a padded utterance see the same neighbourhood
                they see when the utterance is encoded on its own (a padded
                frame is not zero at this point: the pointwise convolution
                and the GLU have already put their biases into it).

        Returns:
            torch.Tensor: Output tensor (#batch, time, channels).

        """
        # exchange the temporal dimension and the feature dimension
        x = x.transpose(1, 2)

        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*channel, dim)
        x = nn.functional.glu(x, dim=1)  # (batch, channel, dim)

        # 1D Depthwise Conv
        x = mask_padded_frames(x, mask_pad)
        x = self.depthwise_conv(x)
        x = self.activation(self.norm(x))

        x = self.pointwise_conv2(x)

        return x.transpose(1, 2)

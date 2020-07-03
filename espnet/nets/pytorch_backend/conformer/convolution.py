#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Johns Hopkins University (Shinji Watanabe)
#                Northwestern Polytechnical University (Pengcheng Guo)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""ConvolutionBlock layer definition."""

from torch import nn


class ConvolutionBlock(nn.Module):
    """ConvolutionBlock layer.

    :param int channels: channels of cnn
    :param int kernel_size: kernerl size of cnn

    """

    def __init__(
        self,
        channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
        activation=nn.ReLU(),
    ):
        """Construct an ConvolutionBlock object."""
        super(ConvolutionBlock, self).__init__()
        self.pad_left = nn.ConstantPad1d((kernel_size - 1, 0), 0)
        self.pointwise_cov1 = nn.Conv1d(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            bias=bias,
        )
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=channels,
            bias=bias,
        )
        self.norm = nn.BatchNorm1d(channels, eps=1e-3, momentum=0.1)
        self.pointwise_cov2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=bias,
        )
        self.act = activation

    def forward(self, x):
        """Compute Covolution Block.

        :param torch.Tensor x: (batch, time, size)
        :return torch.Tensor: convoluted `value` (batch, time, d_model)
        """
        # exchange the temporal dimension and the feature dimension
        # pad the input from (batch, len, dim) to (batch, dim, len+(k-1))
        x = self.pad_left(x.transpose(1, 2))

        # GLU mechanism
        x = self.pointwise_cov1(x)  # (batch, 2*channel, dim)
        x = nn.functional.glu(x, dim=1)  # (batch, channel, dim)

        # 1D Depthwise Conv
        x = self.depthwise_conv(x)
        x = self.act(self.norm(x))

        x = self.pointwise_cov2(x)

        return x.transpose(1, 2)

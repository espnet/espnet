#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Johns Hopkins University (Shinji Watanabe)
#                Northwestern Polytechnical University (Pengcheng Guo)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""ConvolutionModule definition."""

import torch
from torch import nn
import pdb
import logging


class ConvolutionModule(nn.Module):
    """ConvolutionModule in Conformer model.

    Args:
        channels (int): The number of channels of conv layers.
        kernel_size (int): Kernerl size of conv layers.

    """

    def __init__(
        self, channels, kernel_size, block_len=None, activation=nn.ReLU(), bias=True
    ):
        """Construct an ConvolutionModule object."""
        super(ConvolutionModule, self).__init__()
        # kernerl_size should be a odd number for 'SAME' padding
        assert (kernel_size - 1) % 2 == 0
        self.kernel_size = kernel_size
        self.block_len = block_len
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

    def forward(self, x, bl=None):
        """Compute convolution module.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, channels).

        Returns:
            torch.Tensor: Output tensor (#batch, time, channels).

        """
        n_batch, time, dim = x.size()
        if bl is not None:
            self.block_len = bl
        if self.block_len > 0:
            blen = self.block_len
            if time % blen > 0:
                plen = blen - time % blen
                xlen = time + plen
            else:
                plen = 0
                xlen = time
            x = nn.functional.pad(x, (0, 0, blen, plen))
            x = x.transpose(1, 2)  # x(batch, dim, xlen)
            # GLU mechanism
            x = self.pointwise_conv1(x)  # (batch, 2*dim, time)
            x = nn.functional.glu(x, dim=1)  # (batch, dim, time)

            x = x.as_strided(
                (n_batch, int(xlen / blen), blen * 2, dim),
                ((blen + xlen) * dim, dim * blen, dim, 1),
            )

            x = (
                x.contiguous().view(-1, blen * 2, dim).transpose(1, 2)
            )  # (batch, dim, time)
            x = self.depthwise_conv(x)
            x = x[:, :, blen:]
            x = x.transpose(1, 2).contiguous().view(n_batch, -1, dim)
            x = self.activation(self.norm(x.transpose(1, 2)))
            x = self.pointwise_conv2(x)
            # exchange the temporal dimension and the feature dimension

            return x.transpose(1, 2)[:, :time, :]
        else:
            # exchange the temporal dimension and the feature dimension
            x = x.transpose(1, 2)

            # GLU mechanism
            x = self.pointwise_conv1(x)  # (batch, 2*channel, dim)
            x = nn.functional.glu(x, dim=1)  # (batch, channel, dim)

            # 1D Depthwise Conv
            x = self.depthwise_conv(x)
            x = self.activation(self.norm(x))

            x = self.pointwise_conv2(x)
            return x.transpose(1, 2)

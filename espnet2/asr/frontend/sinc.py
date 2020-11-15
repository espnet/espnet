#!/usr/bin/env python3
# encoding: utf-8
#  2020, Technische Universität München;  Ludwig Kürzinger
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Sinc convolutions for raw audio input."""

from collections import OrderedDict
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.layers.sinc_conv import LogCompression
from espnet2.layers.sinc_conv import SincConv
import humanfriendly
import torch
from typing import Tuple
from typing import Union


class LightweightSincConvs(AbsFrontend):
    """Lightweight Sinc Convolutions.

    Provide a frontend for raw audio input.
    https://arxiv.org/abs/2010.07597
    """

    def __init__(
        self,
        fs: Union[int, str] = 16000,
        in_channels=1,
        out_channels=256,
        activation_type="leakyrelu",
        dropout_type="dropout",
    ):
        """Initialize the module.

        :param fs: Sample rate
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels (for each input channel)
        :param activation_type: Choice of activation function
        :param dropout_type:  Choice of dropout function
        """
        super().__init__()
        if isinstance(fs, str):
            fs = humanfriendly.parse_size(fs)
        self.fs = fs
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation_type = activation_type
        self.dropout_type = dropout_type

        self.choices_dropout = {
            "none": torch.nn.Identity,
            "dropout": torch.nn.Dropout,
            "spatial": SpatialDropout,
            "dropout2d": torch.nn.Dropout2d,
        }

        self.choices_activation = {
            "leakyrelu": torch.nn.LeakyReLU,
            "relu": torch.nn.ReLU,
        }

        # initialization
        self._create_sinc_convs()
        self.init_sinc_convs()
        self.espnet_initialization_fn = self.init_sinc_convs

    def _create_sinc_convs(self):
        blocks = OrderedDict()

        # SincConvBlock
        out_channels = 128
        self.filters = SincConv(
            self.in_channels, out_channels, kernel_size=101, stride=1, fs=self.fs
        )
        block = OrderedDict(
            [
                ("Filters", self.filters),
                ("LogCompression", LogCompression()),
                ("BatchNorm", torch.nn.BatchNorm1d(out_channels, affine=True)),
                ("AvgPool", torch.nn.AvgPool1d(2)),
            ]
        )
        blocks["SincConvBlock"] = torch.nn.Sequential(block)
        in_channels = out_channels

        # First convolutional block, connects the sinc output to the front-end "body"
        out_channels = 128
        blocks["DConvBlock1"] = self.gen_lsc_block(
            in_channels,
            out_channels,
            depthwise_kernel_size=25,
            depthwise_stride=2,
            pointwise_groups=0,
            avgpool=True,
            dropout_probability=0.1,
        )
        in_channels = out_channels

        # Second convolutional block, multiple convolutional layers
        out_channels = self.out_channels
        for layer in [2, 3, 4]:
            blocks[f"DConvBlock{layer}"] = self.gen_lsc_block(
                in_channels, out_channels, depthwise_kernel_size=9, depthwise_stride=1
            )
            in_channels = out_channels

        # Third Convolutional block, acts as coupling to encoder
        out_channels = self.out_channels
        blocks["DConvBlock5"] = self.gen_lsc_block(
            in_channels,
            out_channels,
            depthwise_kernel_size=7,
            depthwise_stride=1,
            pointwise_groups=0,
        )

        self.blocks = torch.nn.Sequential(blocks)

    def gen_lsc_block(
        self,
        in_channels,
        out_channels,
        depthwise_kernel_size=9,
        depthwise_stride=1,
        depthwise_groups=None,
        pointwise_groups=0,
        dropout_probability=0.15,
        avgpool=False,
    ):
        """Generate a block for lightweight Sinc convolutions.

        :param in_channels:  Number of input channels
        :param out_channels:  Number of output channels
        :param depthwise_kernel_size: Kernel size of the depthwise convolution
        :param depthwise_stride: Stride of the depthwise convolution
        :param depthwise_groups: Number of groups of the depthwise convolution
        :param pointwise_groups: Number of groups of the pointwise convolution
        :param dropout_probability: Dropout probability in the block
        :param avgpool: If True, an AvgPool layer is inserted
        :return:
        """
        block = OrderedDict()
        if not depthwise_groups:
            # GCD(in_channels, out_channels) to prevent size mismatches
            depthwise_groups, r = in_channels, out_channels
            while r != 0:
                depthwise_groups, r = depthwise_groups, depthwise_groups % r
        block["depthwise"] = torch.nn.Conv1d(
            in_channels,
            out_channels,
            depthwise_kernel_size,
            depthwise_stride,
            groups=depthwise_groups,
        )
        if pointwise_groups:
            block["pointwise"] = torch.nn.Conv1d(
                out_channels, out_channels, 1, 1, groups=pointwise_groups
            )
        block["activation"] = self.choices_activation[self.activation_type]()
        block["batchnorm"] = torch.nn.BatchNorm1d(out_channels, affine=True)
        if avgpool:
            block["avgpool"] = torch.nn.AvgPool1d(2)
        block["dropout"] = self.choices_dropout[self.dropout_type](dropout_probability)
        return torch.nn.Sequential(block)

    def init_sinc_convs(self):
        """Initialize sinc filters with filterbank values."""
        self.filters.init_filters()
        for block in self.blocks:
            for layer in block:
                if type(layer) == torch.nn.BatchNorm1d and layer.affine:
                    layer.weight.data[:] = 1.0
                    layer.bias.data[:] = 0.0

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward function."""
        # Transform input data:
        #   (B, T, C_in, D_in) -> (B*T, C_in, D_in)
        B, T, C_in, D_in = input.size()
        input_frames = input.view(B * T, C_in, D_in)
        output_frames = self.blocks.forward(input_frames)

        # ---TRANSFORM: (B*T, C_out, D_out) -> (B, T, C_out*D_out)
        _, C_out, D_out = output_frames.size()
        output_frames = output_frames.view(B, T, C_out * D_out)
        return output_frames, input_lengths  # no state in this layer

    def get_odim(self, idim=400):
        """Get output dimension by making one inference.

        The test vector that is used has dimentions (1,T,idim).
        T set to idim without any special reason
        :param idim: input dimension D (sample points within one frame)
        :return: output size
        """
        in_test = torch.zeros((1, idim, idim))
        out, _ = self.forward(in_test, [idim])
        return out.size(2)

    def output_size(self) -> int:
        """Get the output size."""
        return self.out_channels * self.in_channels


class SpatialDropout(torch.nn.Module):
    """Spatial dropout module.

    Apply dropout to full channels on tensors of input (B, C, D)
    """

    def __init__(self, dropout_probability=0.15, shape=None):
        """Initialize.

        :param dropout_probability: Dropout probability
        :param shape: Shape as tuple or list
        """
        super().__init__()
        if shape is None:
            shape = (0, 2, 1)
        self.dropout = torch.nn.Dropout2d(dropout_probability)
        self.shape = (shape,)

    def forward(self, x):
        """Forward of spatial dropout module."""
        y = x.permute(*self.shape)
        y = self.dropout(y)
        return y.permute(*self.shape)

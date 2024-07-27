#!/usr/bin/env python3
#  2020, Technische Universität München;  Ludwig Kürzinger
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Sinc convolutions for raw audio input."""

from collections import OrderedDict
from typing import Optional, Tuple, Union

import humanfriendly
import torch
from typeguard import typechecked

from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.layers.sinc_conv import LogCompression, SincConv


class LightweightSincConvs(AbsPreEncoder):
    """
        Lightweight Sinc Convolutions for end-to-end speech recognition from raw audio.

    This class implements the Lightweight Sinc Convolutions as described in
    "Lightweight End-to-End Speech Recognition from Raw Audio Data Using
    Sinc-Convolutions" by Kürzinger et al. It processes raw audio input
    instead of using precomputed features, serving as a pre-encoder in the
    speech recognition pipeline.

    To use this module, set it as the pre-encoder with `preencoder: sinc`
    and use the sliding window frontend with `frontend: sliding_window` in your
    YAML configuration file.

    The process flow should be:
    Frontend (SlidingWindow) -> SpecAug -> Normalization ->
    Pre-encoder (LightweightSincConvs) -> Encoder -> Decoder

    This method performs data augmentation in the time domain, as opposed to
    the spectral domain in the default frontend.

    Attributes:
        fs (int): Sample rate of the input audio.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels (for each input channel).
        activation_type (str): Type of activation function used.
        dropout_type (str): Type of dropout function used.
        windowing_type (str): Type of windowing function used.
        scale_type (str): Type of filter-bank initialization scale.

    Note:
        Use `plot_sinc_filters.py` to visualize the learned Sinc filters.
    """

    @typechecked
    def __init__(
        self,
        fs: Union[int, str, float] = 16000,
        in_channels: int = 1,
        out_channels: int = 256,
        activation_type: str = "leakyrelu",
        dropout_type: str = "dropout",
        windowing_type: str = "hamming",
        scale_type: str = "mel",
    ):
        """Initialize the module.

        Args:
            fs: Sample rate.
            in_channels: Number of input channels.
            out_channels: Number of output channels (for each input channel).
            activation_type: Choice of activation function.
            dropout_type: Choice of dropout function.
            windowing_type: Choice of windowing function.
            scale_type:  Choice of filter-bank initialization scale.
        """
        super().__init__()
        if isinstance(fs, str):
            fs = humanfriendly.parse_size(fs)
        self.fs = fs
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation_type = activation_type
        self.dropout_type = dropout_type
        self.windowing_type = windowing_type
        self.scale_type = scale_type

        self.choices_dropout = {
            "dropout": torch.nn.Dropout,
            "spatial": SpatialDropout,
            "dropout2d": torch.nn.Dropout2d,
        }
        if dropout_type not in self.choices_dropout:
            raise NotImplementedError(
                f"Dropout type has to be one of "
                f"{list(self.choices_dropout.keys())}",
            )

        self.choices_activation = {
            "leakyrelu": torch.nn.LeakyReLU,
            "relu": torch.nn.ReLU,
        }
        if activation_type not in self.choices_activation:
            raise NotImplementedError(
                f"Activation type has to be one of "
                f"{list(self.choices_activation.keys())}",
            )

        # initialization
        self._create_sinc_convs()
        # Sinc filters require custom initialization
        self.espnet_initialization_fn()

    def _create_sinc_convs(self):
        blocks = OrderedDict()

        # SincConvBlock
        out_channels = 128
        self.filters = SincConv(
            self.in_channels,
            out_channels,
            kernel_size=101,
            stride=1,
            fs=self.fs,
            window_func=self.windowing_type,
            scale_type=self.scale_type,
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
        in_channels: int,
        out_channels: int,
        depthwise_kernel_size: int = 9,
        depthwise_stride: int = 1,
        depthwise_groups=None,
        pointwise_groups=0,
        dropout_probability: float = 0.15,
        avgpool=False,
    ):
        """
                Generate a convolutional block for Lightweight Sinc convolutions.

        Each block consists of either a depthwise or a depthwise-separable
        convolution together with dropout, (batch-)normalization layer, and
        an optional average-pooling layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            depthwise_kernel_size (int): Kernel size of the depthwise convolution. Defaults to 9.
            depthwise_stride (int): Stride of the depthwise convolution. Defaults to 1.
            depthwise_groups (int, optional): Number of groups of the depthwise convolution.
                If None, it's set to the GCD of in_channels and out_channels.
            pointwise_groups (int): Number of groups of the pointwise convolution. Defaults to 0.
            dropout_probability (float): Dropout probability in the block. Defaults to 0.15.
            avgpool (bool): If True, an AvgPool layer is inserted. Defaults to False.

        Returns:
            torch.nn.Sequential: Neural network building block for Lightweight Sinc convolutions.

        Note:
            The block structure adapts based on the provided parameters, allowing for
            flexible configuration of the convolutional layers.
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

    def espnet_initialization_fn(self):
        """
                Initialize sinc filters and batch normalization layers.

        This method initializes the sinc filters with filterbank values and sets
        the initial weights and biases for batch normalization layers in the network.

        The initialization process includes:
        1. Initializing the sinc filters using the `init_filters` method.
        2. Setting the weight data of all BatchNorm1d layers to 1.0.
        3. Setting the bias data of all BatchNorm1d layers to 0.0.

        Note:
            This initialization is specific to the Lightweight Sinc Convolutions
            architecture and helps in achieving better initial performance.
        """
        self.filters.init_filters()
        for block in self.blocks:
            for layer in block:
                if type(layer) is torch.nn.BatchNorm1d and layer.affine:
                    layer.weight.data[:] = 1.0
                    layer.bias.data[:] = 0.0

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
                Apply Lightweight Sinc Convolutions to the input tensor.

        Args:
            input (torch.Tensor): Input tensor of shape (B, T, C_in, D_in), where:
                B is the batch size,
                T is the time dimension,
                C_in is the number of input channels,
                D_in is the input feature dimension.
            input_lengths (torch.Tensor): Tensor containing the lengths of each sequence in the batch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - output_frames (torch.Tensor): Output tensor of shape (B, T, C_out*D_out), where:
                    C_out is the number of output channels,
                    D_out is the output feature dimension.
                - input_lengths (torch.Tensor): The input lengths tensor (unchanged).

        Note:
            The current module structure only handles D_in=400, resulting in D_out=1.
            For the multichannel case, C_out is the number of out_channels given at
            initialization multiplied by C_in.

        Example:
            >>> input_tensor = torch.randn(32, 1000, 1, 400)  # (B, T, C_in, D_in)
            >>> input_lengths = torch.full((32,), 1000)
            >>> output, output_lengths = self.forward(input_tensor, input_lengths)
            >>> print(output.shape)
            torch.Size([32, 1000, 256])  # Assuming out_channels=256
        """
        # Transform input data:
        #   (B, T, C_in, D_in) -> (B*T, C_in, D_in)
        B, T, C_in, D_in = input.size()
        input_frames = input.view(B * T, C_in, D_in)
        output_frames = self.blocks.forward(input_frames)

        # ---TRANSFORM: (B*T, C_out, D_out) -> (B, T, C_out*D_out)
        _, C_out, D_out = output_frames.size()
        output_frames = output_frames.view(B, T, C_out * D_out)
        return output_frames, input_lengths  # no state in this layer

    def output_size(self) -> int:
        """
                Get the output size of the Lightweight Sinc Convolutions.

        Returns:
            int: The total number of output features, calculated as the product
            of the number of output channels and the number of input channels.

        Note:
            This method is useful for determining the size of the output tensor
            produced by the Lightweight Sinc Convolutions, which can be helpful
            when connecting this module to subsequent layers in the network.
        """
        return self.out_channels * self.in_channels


class SpatialDropout(torch.nn.Module):
    """
        Spatial dropout module for applying dropout to full channels.

    This module applies dropout to entire channels on tensors of input shape (B, C, D),
    where B is the batch size, C is the number of channels, and D is the feature dimension.
    It's designed to maintain spatial coherence by dropping out entire feature maps
    instead of individual elements.

    Attributes:
        dropout (torch.nn.Dropout2d): The underlying dropout layer.
        shape (tuple): The shape used for permuting the input tensor.

    Note:
        This implementation is particularly useful for 1D convolutional neural networks
        where you want to drop entire channels rather than random elements.
    """

    @typechecked
    def __init__(
        self,
        dropout_probability: float = 0.15,
        shape: Optional[Union[tuple, list]] = None,
    ):
        """Initialize.

        Args:
            dropout_probability: Dropout probability.
            shape (tuple, list): Shape of input tensors.
        """
        super().__init__()
        if shape is None:
            shape = (0, 2, 1)
        self.dropout = torch.nn.Dropout2d(dropout_probability)
        self.shape = (shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
                Apply spatial dropout to the input tensor.

        This method applies dropout to entire channels of the input tensor.
        It first permutes the input tensor according to the specified shape,
        applies dropout, and then permutes the tensor back to its original shape.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, D), where B is the batch size,
                              C is the number of channels, and D is the feature dimension.

        Returns:
            torch.Tensor: Output tensor with spatial dropout applied, maintaining the
                          same shape as the input tensor.

        Note:
            The dropout is applied to entire channels, preserving spatial coherence
            within each channel.
        """
        y = x.permute(*self.shape)
        y = self.dropout(y)
        return y.permute(*self.shape)

# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""StyleMelGAN's TADEResBlock Modules.

This code is modified from https://github.com/kan-bayashi/ParallelWaveGAN.

"""

from functools import partial

import torch


class TADELayer(torch.nn.Module):
    """
        StyleMelGAN's TADEResBlock Modules.

    This code is modified from https://github.com/kan-bayashi/ParallelWaveGAN.

    Attributes:
        in_channels (int): Number of input channels.
        aux_channels (int): Number of auxiliary channels.
        kernel_size (int): Kernel size for convolution layers.
        bias (bool): Whether to use bias in convolution layers.
        upsample_factor (int): Factor by which to upsample the input.
        upsample_mode (str): Mode of upsampling (e.g., 'nearest').

    Examples:
        # Create an instance of TADELayer
        tade_layer = TADELayer(in_channels=64, aux_channels=80)

        # Forward pass through TADELayer
        output, aux = tade_layer(torch.randn(1, 64, 100), torch.randn(1, 80, 50))

        # Create an instance of TADEResBlock
        tade_res_block = TADEResBlock(in_channels=64, aux_channels=80)

        # Forward pass through TADEResBlock
        output, aux = tade_res_block(torch.randn(1, 64, 100), torch.randn(1, 80, 50))
    """

    def __init__(
        self,
        in_channels: int = 64,
        aux_channels: int = 80,
        kernel_size: int = 9,
        bias: bool = True,
        upsample_factor: int = 2,
        upsample_mode: str = "nearest",
    ):
        """Initilize TADELayer module.

        Args:
            in_channels (int): Number of input channles.
            aux_channels (int): Number of auxirialy channles.
            kernel_size (int): Kernel size.
            bias (bool): Whether to use bias parameter in conv.
            upsample_factor (int): Upsample factor.
            upsample_mode (str): Upsample mode.

        """
        super().__init__()
        self.norm = torch.nn.InstanceNorm1d(in_channels)
        self.aux_conv = torch.nn.Sequential(
            torch.nn.Conv1d(
                aux_channels,
                in_channels,
                kernel_size,
                1,
                bias=bias,
                padding=(kernel_size - 1) // 2,
            ),
            # NOTE(kan-bayashi): Use non-linear activation?
        )
        self.gated_conv = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels,
                in_channels * 2,
                kernel_size,
                1,
                bias=bias,
                padding=(kernel_size - 1) // 2,
            ),
            # NOTE(kan-bayashi): Use non-linear activation?
        )
        self.upsample = torch.nn.Upsample(
            scale_factor=upsample_factor, mode=upsample_mode
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, in_channels, T).
            c (Tensor): Auxiliary input tensor (B, aux_channels, T').

        Returns:
            Tensor: Output tensor (B, in_channels, T * in_upsample_factor).
            Tensor: Upsampled aux tensor (B, in_channels, T * aux_upsample_factor).

        """
        x = self.norm(x)
        c = self.upsample(c)
        c = self.aux_conv(c)
        cg = self.gated_conv(c)
        cg1, cg2 = cg.split(cg.size(1) // 2, dim=1)
        # NOTE(kan-bayashi): Use upsample for noise input here?
        y = cg1 * self.upsample(x) + cg2
        # NOTE(kan-bayashi): Return upsampled aux here?
        return y, c


class TADEResBlock(torch.nn.Module):
    """
        TADEResBlock module for the StyleMelGAN architecture.

    This module implements a residual block that incorporates two TADE layers for
    style-based mel-spectrogram generation. It processes input and auxiliary
    tensors to produce an output tensor while applying gated convolutions and
    upsampling. This design is adapted from the original ParallelWaveGAN code.

    Attributes:
        tade1 (TADELayer): The first TADE layer in the residual block.
        gated_conv1 (Conv1d): The first gated convolution layer.
        tade2 (TADELayer): The second TADE layer in the residual block.
        gated_conv2 (Conv1d): The second gated convolution layer.
        upsample (Upsample): Upsampling layer for the output tensor.
        gated_function (Callable): The gating function applied in the block.

    Args:
        in_channels (int): Number of input channels. Default is 64.
        aux_channels (int): Number of auxiliary channels. Default is 80.
        kernel_size (int): Size of the convolutional kernel. Default is 9.
        dilation (int): Dilation rate for the second gated convolution. Default is 2.
        bias (bool): Whether to use a bias parameter in convolutions. Default is True.
        upsample_factor (int): Factor by which to upsample the output. Default is 2.
        upsample_mode (str): Mode of upsampling (e.g., 'nearest'). Default is 'nearest'.
        gated_function (str): Type of gated function ('softmax' or 'sigmoid').
            Default is 'softmax'.

    Raises:
        ValueError: If an unsupported gated_function type is provided.

    Examples:
        >>> res_block = TADEResBlock(in_channels=64, aux_channels=80)
        >>> x = torch.randn(1, 64, 100)  # Input tensor
        >>> c = torch.randn(1, 80, 50)    # Auxiliary tensor
        >>> output, aux = res_block(x, c)
        >>> print(output.shape)  # Output shape will be (1, 64, 200)
    """

    def __init__(
        self,
        in_channels: int = 64,
        aux_channels: int = 80,
        kernel_size: int = 9,
        dilation: int = 2,
        bias: bool = True,
        upsample_factor: int = 2,
        upsample_mode: str = "nearest",
        gated_function: str = "softmax",
    ):
        """Initialize TADEResBlock module.

        Args:
            in_channels (int): Number of input channles.
            aux_channels (int): Number of auxirialy channles.
            kernel_size (int): Kernel size.
            bias (bool): Whether to use bias parameter in conv.
            upsample_factor (int): Upsample factor.
            upsample_mode (str): Upsample mode.
            gated_function (str): Gated function type (softmax of sigmoid).

        """
        super().__init__()
        self.tade1 = TADELayer(
            in_channels=in_channels,
            aux_channels=aux_channels,
            kernel_size=kernel_size,
            bias=bias,
            # NOTE(kan-bayashi): Use upsample in the first TADE layer?
            upsample_factor=1,
            upsample_mode=upsample_mode,
        )
        self.gated_conv1 = torch.nn.Conv1d(
            in_channels,
            in_channels * 2,
            kernel_size,
            1,
            bias=bias,
            padding=(kernel_size - 1) // 2,
        )
        self.tade2 = TADELayer(
            in_channels=in_channels,
            aux_channels=in_channels,
            kernel_size=kernel_size,
            bias=bias,
            upsample_factor=upsample_factor,
            upsample_mode=upsample_mode,
        )
        self.gated_conv2 = torch.nn.Conv1d(
            in_channels,
            in_channels * 2,
            kernel_size,
            1,
            bias=bias,
            dilation=dilation,
            padding=(kernel_size - 1) // 2 * dilation,
        )
        self.upsample = torch.nn.Upsample(
            scale_factor=upsample_factor, mode=upsample_mode
        )
        if gated_function == "softmax":
            self.gated_function = partial(torch.softmax, dim=1)
        elif gated_function == "sigmoid":
            self.gated_function = torch.sigmoid
        else:
            raise ValueError(f"{gated_function} is not supported.")

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, in_channels, T).
            c (Tensor): Auxiliary input tensor (B, aux_channels, T').

        Returns:
            Tensor: Output tensor (B, in_channels, T * in_upsample_factor).
            Tensor: Upsampled auxirialy tensor (B, in_channels, T * in_upsample_factor).

        """
        residual = x

        x, c = self.tade1(x, c)
        x = self.gated_conv1(x)
        xa, xb = x.split(x.size(1) // 2, dim=1)
        x = self.gated_function(xa) * torch.tanh(xb)

        x, c = self.tade2(x, c)
        x = self.gated_conv2(x)
        xa, xb = x.split(x.size(1) // 2, dim=1)
        x = self.gated_function(xa) * torch.tanh(xb)

        # NOTE(kan-bayashi): Return upsampled aux here?
        return self.upsample(residual) + x, c

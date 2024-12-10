# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Residual block modules.

This code is modified from https://github.com/kan-bayashi/ParallelWaveGAN.

"""

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


class Conv1d(torch.nn.Conv1d):
    """
        Conv1d module with customized initialization.

    This class is a subclass of `torch.nn.Conv1d` and provides a customized
    initialization for the convolutional layer weights using Kaiming normal
    initialization and sets the bias to zero if it exists.

    Attributes:
        weight (Tensor): The learnable weights of the module of shape
            (out_channels, in_channels, kernel_size).
        bias (Tensor, optional): The learnable bias of the module of shape
            (out_channels).

    Args:
        *args: Variable length argument list for the parent `torch.nn.Conv1d`.
        **kwargs: Arbitrary keyword arguments for the parent `torch.nn.Conv1d`.

    Examples:
        >>> conv = Conv1d(in_channels=3, out_channels=2, kernel_size=5)
        >>> print(conv)
    """

    def __init__(self, *args, **kwargs):
        """Initialize Conv1d module."""
        super().__init__(*args, **kwargs)

    def reset_parameters(self):
        """
        Reset parameters of the Conv1d layer.

        This method initializes the weight of the convolutional layer using the
        Kaiming normal initialization, which is suitable for layers followed by
        a ReLU activation function. If the layer has a bias term, it is set to
        zero.

        Attributes:
            weight (torch.Tensor): The weight of the convolutional layer.
            bias (Optional[torch.Tensor]): The bias of the convolutional layer,
                if applicable.

        Raises:
            ValueError: If the weight tensor is not initialized properly.

        Examples:
            >>> conv_layer = Conv1d(in_channels=1, out_channels=2, kernel_size=3)
            >>> conv_layer.reset_parameters()
            >>> print(conv_layer.weight)
            tensor([[...], ...])  # Initialized weights
            >>> print(conv_layer.bias)
            tensor([0.])  # Bias initialized to zero
        """
        torch.nn.init.kaiming_normal_(self.weight, nonlinearity="relu")
        if self.bias is not None:
            torch.nn.init.constant_(self.bias, 0.0)


class Conv1d1x1(Conv1d):
    """
        1x1 Conv1d with customized initialization.

    This class implements a 1x1 convolutional layer that extends the Conv1d class
    with a specific initialization method for its weights.

    Attributes:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        bias (bool): Whether to include a bias term in the convolution.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        bias (bool): Whether to add bias parameter in the convolution.

    Examples:
        >>> conv1d_1x1 = Conv1d1x1(in_channels=16, out_channels=32, bias=True)
        >>> input_tensor = torch.randn(1, 16, 10)  # (batch_size, channels, length)
        >>> output_tensor = conv1d_1x1(input_tensor)
        >>> output_tensor.shape
        torch.Size([1, 32, 10])

    Note:
        This layer is primarily used in the context of neural network architectures
        where 1x1 convolutions are beneficial, such as in residual networks.
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool):
        """Initialize 1x1 Conv1d module."""
        super().__init__(
            in_channels, out_channels, kernel_size=1, padding=0, dilation=1, bias=bias
        )


class ResidualBlock(torch.nn.Module):
    """
        Residual block module in WaveNet.

    This module implements a residual block used in the WaveNet architecture.
    It incorporates convolutional layers with gated activation and allows for
    local and global conditioning. This code is modified from
    https://github.com/kan-bayashi/ParallelWaveGAN.

    Attributes:
        dropout_rate (float): The probability of dropout applied to the input.
        residual_channels (int): Number of channels for the residual connection.
        skip_channels (int): Number of channels for the skip connection.
        scale_residual (bool): Whether to scale the residual outputs.

    Args:
        kernel_size (int): Kernel size of dilation convolution layer.
        residual_channels (int): Number of channels for residual connection.
        gate_channels (int): Number of channels for gating mechanism.
        skip_channels (int): Number of channels for skip connection.
        aux_channels (int): Number of local conditioning channels.
        global_channels (int): Number of global conditioning channels.
        dropout_rate (float): Dropout probability.
        dilation (int): Dilation factor.
        bias (bool): Whether to add bias parameter in convolution layers.
        scale_residual (bool): Whether to scale the residual outputs.

    Examples:
        >>> residual_block = ResidualBlock()
        >>> x = torch.randn(1, 64, 100)  # Example input tensor
        >>> output, skip = residual_block(x)

    Raises:
        AssertionError: If the kernel size is even or gate channels are not even.
    """

    def __init__(
        self,
        kernel_size: int = 3,
        residual_channels: int = 64,
        gate_channels: int = 128,
        skip_channels: int = 64,
        aux_channels: int = 80,
        global_channels: int = -1,
        dropout_rate: float = 0.0,
        dilation: int = 1,
        bias: bool = True,
        scale_residual: bool = False,
    ):
        """Initialize ResidualBlock module.

        Args:
            kernel_size (int): Kernel size of dilation convolution layer.
            residual_channels (int): Number of channels for residual connection.
            skip_channels (int): Number of channels for skip connection.
            aux_channels (int): Number of local conditioning channels.
            dropout (float): Dropout probability.
            dilation (int): Dilation factor.
            bias (bool): Whether to add bias parameter in convolution layers.
            scale_residual (bool): Whether to scale the residual outputs.

        """
        super().__init__()
        self.dropout_rate = dropout_rate
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.scale_residual = scale_residual

        # check
        assert (kernel_size - 1) % 2 == 0, "Not support even number kernel size."
        assert gate_channels % 2 == 0

        # dilation conv
        padding = (kernel_size - 1) // 2 * dilation
        self.conv = Conv1d(
            residual_channels,
            gate_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

        # local conditioning
        if aux_channels > 0:
            self.conv1x1_aux = Conv1d1x1(aux_channels, gate_channels, bias=False)
        else:
            self.conv1x1_aux = None

        # global conditioning
        if global_channels > 0:
            self.conv1x1_glo = Conv1d1x1(global_channels, gate_channels, bias=False)
        else:
            self.conv1x1_glo = None

        # conv output is split into two groups
        gate_out_channels = gate_channels // 2

        # NOTE(kan-bayashi): concat two convs into a single conv for the efficiency
        #   (integrate res 1x1 + skip 1x1 convs)
        self.conv1x1_out = Conv1d1x1(
            gate_out_channels, residual_channels + skip_channels, bias=bias
        )

    def forward(
        self,
        x: torch.Tensor,
        x_mask: Optional[torch.Tensor] = None,
        c: Optional[torch.Tensor] = None,
        g: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
                Calculate forward propagation through the ResidualBlock.

        This method computes the forward pass of the residual block, taking into
        account the input tensor, optional local and global conditioning tensors,
        and an optional mask tensor for attention.

        Args:
            x (Tensor): Input tensor of shape (B, residual_channels, T).
            x_mask (Optional[torch.Tensor]): Mask tensor of shape (B, 1, T).
                Used to zero out certain parts of the output.
            c (Optional[Tensor]): Local conditioning tensor of shape (B, aux_channels, T).
            g (Optional[Tensor]): Global conditioning tensor of shape (B, global_channels, 1).

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing:
                - Output tensor for residual connection of shape
                  (B, residual_channels, T).
                - Output tensor for skip connection of shape
                  (B, skip_channels, T).

        Examples:
            >>> residual_block = ResidualBlock()
            >>> x = torch.randn(1, 64, 100)  # Example input tensor
            >>> output_residual, output_skip = residual_block(x)
            >>> print(output_residual.shape)  # Should be (1, 64, 100)
            >>> print(output_skip.shape)       # Should be (1, 64, 100)
        """
        residual = x
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv(x)

        # split into two part for gated activation
        splitdim = 1
        xa, xb = x.split(x.size(splitdim) // 2, dim=splitdim)

        # local conditioning
        if c is not None:
            c = self.conv1x1_aux(c)
            ca, cb = c.split(c.size(splitdim) // 2, dim=splitdim)
            xa, xb = xa + ca, xb + cb

        # global conditioning
        if g is not None:
            g = self.conv1x1_glo(g)
            ga, gb = g.split(g.size(splitdim) // 2, dim=splitdim)
            xa, xb = xa + ga, xb + gb

        x = torch.tanh(xa) * torch.sigmoid(xb)

        # residual + skip 1x1 conv
        x = self.conv1x1_out(x)
        if x_mask is not None:
            x = x * x_mask

        # split integrated conv results
        x, s = x.split([self.residual_channels, self.skip_channels], dim=1)

        # for residual connection
        x = x + residual
        if self.scale_residual:
            x = x * math.sqrt(0.5)

        return x, s

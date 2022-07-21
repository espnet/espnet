# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Basic Flow modules used in VITS.

This code is based on https://github.com/jaywalnut310/vits.

"""

import math
from typing import Optional, Tuple, Union

import torch

from espnet2.gan_tts.vits.transform import piecewise_rational_quadratic_transform


class FlipFlow(torch.nn.Module):
    """Flip flow module."""

    def forward(
        self, x: torch.Tensor, *args, inverse: bool = False, **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, channels, T).
            inverse (bool): Whether to inverse the flow.

        Returns:
            Tensor: Flipped tensor (B, channels, T).
            Tensor: Log-determinant tensor for NLL (B,) if not inverse.

        """
        x = torch.flip(x, [1])
        if not inverse:
            logdet = x.new_zeros(x.size(0))
            return x, logdet
        else:
            return x


class LogFlow(torch.nn.Module):
    """Log flow module."""

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        inverse: bool = False,
        eps: float = 1e-5,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, channels, T).
            x_mask (Tensor): Mask tensor (B, 1, T).
            inverse (bool): Whether to inverse the flow.
            eps (float): Epsilon for log.

        Returns:
            Tensor: Output tensor (B, channels, T).
            Tensor: Log-determinant tensor for NLL (B,) if not inverse.

        """
        if not inverse:
            y = torch.log(torch.clamp_min(x, eps)) * x_mask
            logdet = torch.sum(-y, [1, 2])
            return y, logdet
        else:
            x = torch.exp(x) * x_mask
            return x


class ElementwiseAffineFlow(torch.nn.Module):
    """Elementwise affine flow module."""

    def __init__(self, channels: int):
        """Initialize ElementwiseAffineFlow module.

        Args:
            channels (int): Number of channels.

        """
        super().__init__()
        self.channels = channels
        self.register_parameter("m", torch.nn.Parameter(torch.zeros(channels, 1)))
        self.register_parameter("logs", torch.nn.Parameter(torch.zeros(channels, 1)))

    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor, inverse: bool = False, **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, channels, T).
            x_lengths (Tensor): Length tensor (B,).
            inverse (bool): Whether to inverse the flow.

        Returns:
            Tensor: Output tensor (B, channels, T).
            Tensor: Log-determinant tensor for NLL (B,) if not inverse.

        """
        if not inverse:
            y = self.m + torch.exp(self.logs) * x
            y = y * x_mask
            logdet = torch.sum(self.logs * x_mask, [1, 2])
            return y, logdet
        else:
            x = (x - self.m) * torch.exp(-self.logs) * x_mask
            return x


class Transpose(torch.nn.Module):
    """Transpose module for torch.nn.Sequential()."""

    def __init__(self, dim1: int, dim2: int):
        """Initialize Transpose module."""
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transpose."""
        return x.transpose(self.dim1, self.dim2)


class DilatedDepthSeparableConv(torch.nn.Module):
    """Dilated depth-separable conv module."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        layers: int,
        dropout_rate: float = 0.0,
        eps: float = 1e-5,
    ):
        """Initialize DilatedDepthSeparableConv module.

        Args:
            channels (int): Number of channels.
            kernel_size (int): Kernel size.
            layers (int): Number of layers.
            dropout_rate (float): Dropout rate.
            eps (float): Epsilon for layer norm.

        """
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for i in range(layers):
            dilation = kernel_size ** i
            padding = (kernel_size * dilation - dilation) // 2
            self.convs += [
                torch.nn.Sequential(
                    torch.nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        groups=channels,
                        dilation=dilation,
                        padding=padding,
                    ),
                    Transpose(1, 2),
                    torch.nn.LayerNorm(channels, eps=eps, elementwise_affine=True,),
                    Transpose(1, 2),
                    torch.nn.GELU(),
                    torch.nn.Conv1d(channels, channels, 1,),
                    Transpose(1, 2),
                    torch.nn.LayerNorm(channels, eps=eps, elementwise_affine=True,),
                    Transpose(1, 2),
                    torch.nn.GELU(),
                    torch.nn.Dropout(dropout_rate),
                )
            ]

    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor, g: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, in_channels, T).
            x_mask (Tensor): Mask tensor (B, 1, T).
            g (Optional[Tensor]): Global conditioning tensor (B, global_channels, 1).

        Returns:
            Tensor: Output tensor (B, channels, T).

        """
        if g is not None:
            x = x + g
        for f in self.convs:
            y = f(x * x_mask)
            x = x + y
        return x * x_mask


class ConvFlow(torch.nn.Module):
    """Convolutional flow module."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int,
        layers: int,
        bins: int = 10,
        tail_bound: float = 5.0,
    ):
        """Initialize ConvFlow module.

        Args:
            in_channels (int): Number of input channels.
            hidden_channels (int): Number of hidden channels.
            kernel_size (int): Kernel size.
            layers (int): Number of layers.
            bins (int): Number of bins.
            tail_bound (float): Tail bound value.

        """
        super().__init__()
        self.half_channels = in_channels // 2
        self.hidden_channels = hidden_channels
        self.bins = bins
        self.tail_bound = tail_bound

        self.input_conv = torch.nn.Conv1d(self.half_channels, hidden_channels, 1,)
        self.dds_conv = DilatedDepthSeparableConv(
            hidden_channels, kernel_size, layers, dropout_rate=0.0,
        )
        self.proj = torch.nn.Conv1d(
            hidden_channels, self.half_channels * (bins * 3 - 1), 1,
        )
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        inverse: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, channels, T).
            x_mask (Tensor): Mask tensor (B,).
            g (Optional[Tensor]): Global conditioning tensor (B, channels, 1).
            inverse (bool): Whether to inverse the flow.

        Returns:
            Tensor: Output tensor (B, channels, T).
            Tensor: Log-determinant tensor for NLL (B,) if not inverse.

        """
        xa, xb = x.split(x.size(1) // 2, 1)
        h = self.input_conv(xa)
        h = self.dds_conv(h, x_mask, g=g)
        h = self.proj(h) * x_mask  # (B, half_channels * (bins * 3 - 1), T)

        b, c, t = xa.shape
        # (B, half_channels, bins * 3 - 1, T) -> (B, half_channels, T, bins * 3 - 1)
        h = h.reshape(b, c, -1, t).permute(0, 1, 3, 2)

        # TODO(kan-bayashi): Understand this calculation
        denom = math.sqrt(self.hidden_channels)
        unnorm_widths = h[..., : self.bins] / denom
        unnorm_heights = h[..., self.bins : 2 * self.bins] / denom
        unnorm_derivatives = h[..., 2 * self.bins :]
        xb, logdet_abs = piecewise_rational_quadratic_transform(
            xb,
            unnorm_widths,
            unnorm_heights,
            unnorm_derivatives,
            inverse=inverse,
            tails="linear",
            tail_bound=self.tail_bound,
        )
        x = torch.cat([xa, xb], 1) * x_mask
        logdet = torch.sum(logdet_abs * x_mask, [1, 2])
        if not inverse:
            return x, logdet
        else:
            return x

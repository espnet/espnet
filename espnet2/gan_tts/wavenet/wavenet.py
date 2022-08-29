# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""WaveNet modules.

This code is modified from https://github.com/kan-bayashi/ParallelWaveGAN.

"""

import logging
import math
from typing import Optional

import torch

from espnet2.gan_tts.wavenet.residual_block import Conv1d1x1, ResidualBlock


class WaveNet(torch.nn.Module):
    """WaveNet with global conditioning."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 3,
        layers: int = 30,
        stacks: int = 3,
        base_dilation: int = 2,
        residual_channels: int = 64,
        aux_channels: int = -1,
        gate_channels: int = 128,
        skip_channels: int = 64,
        global_channels: int = -1,
        dropout_rate: float = 0.0,
        bias: bool = True,
        use_weight_norm: bool = True,
        use_first_conv: bool = False,
        use_last_conv: bool = False,
        scale_residual: bool = False,
        scale_skip_connect: bool = False,
    ):
        """Initialize WaveNet module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Kernel size of dilated convolution.
            layers (int): Number of residual block layers.
            stacks (int): Number of stacks i.e., dilation cycles.
            base_dilation (int): Base dilation factor.
            residual_channels (int): Number of channels in residual conv.
            gate_channels (int):  Number of channels in gated conv.
            skip_channels (int): Number of channels in skip conv.
            aux_channels (int): Number of channels for local conditioning feature.
            global_channels (int): Number of channels for global conditioning feature.
            dropout_rate (float): Dropout rate. 0.0 means no dropout applied.
            bias (bool): Whether to use bias parameter in conv layer.
            use_weight_norm (bool): Whether to use weight norm. If set to true, it will
                be applied to all of the conv layers.
            use_first_conv (bool): Whether to use the first conv layers.
            use_last_conv (bool): Whether to use the last conv layers.
            scale_residual (bool): Whether to scale the residual outputs.
            scale_skip_connect (bool): Whether to scale the skip connection outputs.

        """
        super().__init__()
        self.layers = layers
        self.stacks = stacks
        self.kernel_size = kernel_size
        self.base_dilation = base_dilation
        self.use_first_conv = use_first_conv
        self.use_last_conv = use_last_conv
        self.scale_skip_connect = scale_skip_connect

        # check the number of layers and stacks
        assert layers % stacks == 0
        layers_per_stack = layers // stacks

        # define first convolution
        if self.use_first_conv:
            self.first_conv = Conv1d1x1(in_channels, residual_channels, bias=True)

        # define residual blocks
        self.conv_layers = torch.nn.ModuleList()
        for layer in range(layers):
            dilation = base_dilation ** (layer % layers_per_stack)
            conv = ResidualBlock(
                kernel_size=kernel_size,
                residual_channels=residual_channels,
                gate_channels=gate_channels,
                skip_channels=skip_channels,
                aux_channels=aux_channels,
                global_channels=global_channels,
                dilation=dilation,
                dropout_rate=dropout_rate,
                bias=bias,
                scale_residual=scale_residual,
            )
            self.conv_layers += [conv]

        # define output layers
        if self.use_last_conv:
            self.last_conv = torch.nn.Sequential(
                torch.nn.ReLU(inplace=True),
                Conv1d1x1(skip_channels, skip_channels, bias=True),
                torch.nn.ReLU(inplace=True),
                Conv1d1x1(skip_channels, out_channels, bias=True),
            )

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

    def forward(
        self,
        x: torch.Tensor,
        x_mask: Optional[torch.Tensor] = None,
        c: Optional[torch.Tensor] = None,
        g: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T) if use_first_conv else
                (B, residual_channels, T).
            x_mask (Optional[Tensor]): Mask tensor (B, 1, T).
            c (Optional[Tensor]): Local conditioning features (B, aux_channels, T).
            g (Optional[Tensor]): Global conditioning features (B, global_channels, 1).

        Returns:
            Tensor: Output tensor (B, out_channels, T) if use_last_conv else
                (B, residual_channels, T).

        """
        # encode to hidden representation
        if self.use_first_conv:
            x = self.first_conv(x)

        # residual block
        skips = 0.0
        for f in self.conv_layers:
            x, h = f(x, x_mask=x_mask, c=c, g=g)
            skips = skips + h
        x = skips
        if self.scale_skip_connect:
            x = x * math.sqrt(1.0 / len(self.conv_layers))

        # apply final layers
        if self.use_last_conv:
            x = self.last_conv(x)

        return x

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m: torch.nn.Module):
            try:
                logging.debug(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m: torch.nn.Module):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    @staticmethod
    def _get_receptive_field_size(
        layers: int, stacks: int, kernel_size: int, base_dilation: int,
    ) -> int:
        assert layers % stacks == 0
        layers_per_cycle = layers // stacks
        dilations = [base_dilation ** (i % layers_per_cycle) for i in range(layers)]
        return (kernel_size - 1) * sum(dilations) + 1

    @property
    def receptive_field_size(self) -> int:
        """Return receptive field size."""
        return self._get_receptive_field_size(
            self.layers, self.stacks, self.kernel_size, self.base_dilation
        )

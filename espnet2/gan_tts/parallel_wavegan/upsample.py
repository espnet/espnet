# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Upsampling module.

This code is modified from https://github.com/kan-bayashi/ParallelWaveGAN.

"""

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from espnet2.gan_tts.wavenet.residual_block import Conv1d


class Stretch2d(torch.nn.Module):
    """Stretch2d module."""

    def __init__(self, x_scale: int, y_scale: int, mode: str = "nearest"):
        """Initialize Stretch2d module.

        Args:
            x_scale (int): X scaling factor (Time axis in spectrogram).
            y_scale (int): Y scaling factor (Frequency axis in spectrogram).
            mode (str): Interpolation mode.

        """
        super().__init__()
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, C, F, T).

        Returns:
            Tensor: Interpolated tensor (B, C, F * y_scale, T * x_scale),

        """
        return F.interpolate(
            x, scale_factor=(self.y_scale, self.x_scale), mode=self.mode
        )


class Conv2d(torch.nn.Conv2d):
    """Conv2d module with customized initialization."""

    def __init__(self, *args, **kwargs):
        """Initialize Conv2d module."""
        super().__init__(*args, **kwargs)

    def reset_parameters(self):
        """Reset parameters."""
        self.weight.data.fill_(1.0 / np.prod(self.kernel_size))
        if self.bias is not None:
            torch.nn.init.constant_(self.bias, 0.0)


class UpsampleNetwork(torch.nn.Module):
    """Upsampling network module."""

    def __init__(
        self,
        upsample_scales: List[int],
        nonlinear_activation: Optional[str] = None,
        nonlinear_activation_params: Dict[str, Any] = {},
        interpolate_mode: str = "nearest",
        freq_axis_kernel_size: int = 1,
    ):
        """Initialize UpsampleNetwork module.

        Args:
            upsample_scales (List[int]): List of upsampling scales.
            nonlinear_activation (Optional[str]): Activation function name.
            nonlinear_activation_params (Dict[str, Any]): Arguments for the specified
                activation function.
            interpolate_mode (str): Interpolation mode.
            freq_axis_kernel_size (int): Kernel size in the direction of frequency axis.

        """
        super().__init__()
        self.up_layers = torch.nn.ModuleList()
        for scale in upsample_scales:
            # interpolation layer
            stretch = Stretch2d(scale, 1, interpolate_mode)
            self.up_layers += [stretch]

            # conv layer
            assert (
                freq_axis_kernel_size - 1
            ) % 2 == 0, "Not support even number freq axis kernel size."
            freq_axis_padding = (freq_axis_kernel_size - 1) // 2
            kernel_size = (freq_axis_kernel_size, scale * 2 + 1)
            padding = (freq_axis_padding, scale)
            conv = Conv2d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)
            self.up_layers += [conv]

            # nonlinear
            if nonlinear_activation is not None:
                nonlinear = getattr(torch.nn, nonlinear_activation)(
                    **nonlinear_activation_params
                )
                self.up_layers += [nonlinear]

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        """Calculate forward propagation.

        Args:
            c : Input tensor (B, C, T_feats).

        Returns:
            Tensor: Upsampled tensor (B, C, T_wav).

        """
        c = c.unsqueeze(1)  # (B, 1, C, T)
        for f in self.up_layers:
            c = f(c)
        return c.squeeze(1)  # (B, C, T')


class ConvInUpsampleNetwork(torch.nn.Module):
    """Convolution + upsampling network module."""

    def __init__(
        self,
        upsample_scales: List[int],
        nonlinear_activation: Optional[str] = None,
        nonlinear_activation_params: Dict[str, Any] = {},
        interpolate_mode: str = "nearest",
        freq_axis_kernel_size: int = 1,
        aux_channels: int = 80,
        aux_context_window: int = 0,
    ):
        """Initialize ConvInUpsampleNetwork module.

        Args:
            upsample_scales (list): List of upsampling scales.
            nonlinear_activation (Optional[str]): Activation function name.
            nonlinear_activation_params (Dict[str, Any]): Arguments for the specified
                activation function.
            mode (str): Interpolation mode.
            freq_axis_kernel_size (int): Kernel size in the direction of
                frequency axis.
            aux_channels (int): Number of channels of pre-conv layer.
            aux_context_window (int): Context window size of the pre-conv layer.

        """
        super().__init__()
        self.aux_context_window = aux_context_window
        # To capture wide-context information in conditional features
        kernel_size = 2 * aux_context_window + 1
        # NOTE(kan-bayashi): Use pad here, which is not used in parallel_wavegan
        self.pad = torch.nn.ReplicationPad1d(aux_context_window)
        self.conv_in = Conv1d(
            aux_channels, aux_channels, kernel_size=kernel_size, bias=False,
        )
        self.upsample = UpsampleNetwork(
            upsample_scales=upsample_scales,
            nonlinear_activation=nonlinear_activation,
            nonlinear_activation_params=nonlinear_activation_params,
            interpolate_mode=interpolate_mode,
            freq_axis_kernel_size=freq_axis_kernel_size,
        )

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        """Calculate forward propagation.

        Args:
            c (Tensor): Input tensor (B, C, T_feats).

        Returns:
            Tensor: Upsampled tensor (B, C, T_wav),
                where T_wav = T_feats * prod(upsample_scales).

        """
        c = self.conv_in(self.pad(c))
        return self.upsample(c)

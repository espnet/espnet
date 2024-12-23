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
    """
        Stretch2d module.

    This module performs 2D stretching (upsampling) on input tensors, typically used
    in audio processing applications such as spectrogram manipulation.

    The code is modified from https://github.com/kan-bayashi/ParallelWaveGAN.

    Attributes:
        x_scale (int): X scaling factor (Time axis in spectrogram).
        y_scale (int): Y scaling factor (Frequency axis in spectrogram).
        mode (str): Interpolation mode for upsampling.

    Args:
        x_scale (int): X scaling factor (Time axis in spectrogram).
        y_scale (int): Y scaling factor (Frequency axis in spectrogram).
        mode (str): Interpolation mode.

    Returns:
        None

    Examples:
        >>> import torch
        >>> stretch = Stretch2d(x_scale=2, y_scale=3, mode='nearest')
        >>> input_tensor = torch.randn(1, 1, 4, 4)  # (B, C, F, T)
        >>> output_tensor = stretch(input_tensor)
        >>> output_tensor.shape
        torch.Size([1, 1, 12, 8])  # (B, C, F * y_scale, T * x_scale)
    """

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
        """
            Calculate forward propagation.

        This method performs the forward pass of the Stretch2d module by applying
        upsampling to the input tensor using the specified scaling factors and
        interpolation mode.

        Args:
            x (Tensor): Input tensor of shape (B, C, F, T), where:
                B - Batch size
                C - Number of channels
                F - Frequency bins
                T - Time steps

        Returns:
            Tensor: Interpolated tensor of shape (B, C, F * y_scale, T * x_scale),
                where y_scale and x_scale are the scaling factors defined during
                initialization.

        Examples:
            >>> stretch = Stretch2d(x_scale=2, y_scale=3, mode='linear')
            >>> input_tensor = torch.randn(1, 1, 4, 4)  # Example input
            >>> output_tensor = stretch(input_tensor)
            >>> output_tensor.shape
            torch.Size([1, 1, 12, 8])  # Output shape after upsampling
        """
        return F.interpolate(
            x, scale_factor=(self.y_scale, self.x_scale), mode=self.mode
        )


class Conv2d(torch.nn.Conv2d):
    """
    Conv2d module with customized initialization.

    This module extends the standard PyTorch Conv2d layer to include a custom
    parameter initialization method. The weights are initialized to a constant
    value based on the kernel size, and the bias (if present) is initialized to
    zero.

    Attributes:
        weight (torch.Tensor): The learnable weights of the module of shape
            (out_channels, in_channels, kernel_height, kernel_width).
        bias (torch.Tensor or None): The learnable bias of the module, of shape
            (out_channels,) if bias is enabled, otherwise None.

    Args:
        *args: Variable length argument list for Conv2d.
        **kwargs: Keyword arguments for Conv2d.

    Methods:
        reset_parameters: Resets the parameters of the Conv2d layer.

    Examples:
        >>> conv_layer = Conv2d(in_channels=3, out_channels=16, kernel_size=3)
        >>> print(conv_layer.weight)  # Check initialized weights
        >>> print(conv_layer.bias)     # Check initialized bias
    """

    def __init__(self, *args, **kwargs):
        """Initialize Conv2d module."""
        super().__init__(*args, **kwargs)

    def reset_parameters(self):
        """
            Reset parameters.

        This method initializes the weights of the Conv2d module to a constant value
        based on the kernel size, and sets the bias to zero if it exists. This can be
        useful to reset the model's state before training or fine-tuning.

        The weights are filled with the value `1.0 / np.prod(self.kernel_size)`, which
        ensures that the sum of the weights is normalized to 1 across the kernel.

        Note:
            This method is typically called during the initialization of the model or
            before training to ensure that the weights are set to a known state.

        Examples:
            >>> conv_layer = Conv2d(1, 1, kernel_size=(3, 3))
            >>> conv_layer.reset_parameters()
            >>> print(conv_layer.weight.data)  # Should print the initialized weights

        Raises:
            None
        """
        self.weight.data.fill_(1.0 / np.prod(self.kernel_size))
        if self.bias is not None:
            torch.nn.init.constant_(self.bias, 0.0)


class UpsampleNetwork(torch.nn.Module):
    """
    Upsampling network module.

    This module performs upsampling on input tensors through a series of
    interpolation and convolution layers. It allows for customizable
    non-linear activation functions and scaling factors.

    Args:
        upsample_scales (List[int]): List of upsampling scales.
        nonlinear_activation (Optional[str]): Activation function name.
        nonlinear_activation_params (Dict[str, Any]): Arguments for the specified
            activation function.
        interpolate_mode (str): Interpolation mode for upsampling.
        freq_axis_kernel_size (int): Kernel size in the direction of frequency axis.

    Examples:
        >>> upsample_network = UpsampleNetwork(
        ...     upsample_scales=[2, 2],
        ...     nonlinear_activation='ReLU',
        ...     nonlinear_activation_params={},
        ...     interpolate_mode='nearest',
        ...     freq_axis_kernel_size=3
        ... )
        >>> input_tensor = torch.randn(1, 80, 50)  # (B, C, T_feats)
        >>> output_tensor = upsample_network(input_tensor)
        >>> output_tensor.shape
        torch.Size([1, 80, 200])  # (B, C, T_wav)
    """

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
        """
            Calculate forward propagation.

        This method processes the input tensor through a series of upsampling layers
        and returns the upsampled tensor.

        Args:
            c (Tensor): Input tensor of shape (B, C, T_feats), where:
                - B: Batch size
                - C: Number of channels
                - T_feats: Number of feature dimensions

        Returns:
            Tensor: Upsampled tensor of shape (B, C, T_wav), where:
                - T_wav = T_feats * prod(upsample_scales), representing the
                  total length after upsampling.

        Examples:
            >>> model = UpsampleNetwork(upsample_scales=[2, 2])
            >>> input_tensor = torch.randn(1, 10, 5)  # Example input
            >>> output_tensor = model(input_tensor)
            >>> output_tensor.shape
            torch.Size([1, 10, 20])  # Example output shape after upsampling
        """
        c = c.unsqueeze(1)  # (B, 1, C, T)
        for f in self.up_layers:
            c = f(c)
        return c.squeeze(1)  # (B, C, T')


class ConvInUpsampleNetwork(torch.nn.Module):
    """
        Convolution + upsampling network module.

    This module combines a convolutional layer with an upsampling network to
    process input tensors, typically for tasks like audio synthesis.

    Args:
        upsample_scales (List[int]): List of upsampling scales.
        nonlinear_activation (Optional[str]): Activation function name.
        nonlinear_activation_params (Dict[str, Any]): Arguments for the specified
            activation function.
        interpolate_mode (str): Interpolation mode.
        freq_axis_kernel_size (int): Kernel size in the direction of
            frequency axis.
        aux_channels (int): Number of channels of pre-conv layer.
        aux_context_window (int): Context window size of the pre-conv layer.

    Examples:
        >>> model = ConvInUpsampleNetwork(
        ...     upsample_scales=[2, 2],
        ...     nonlinear_activation='ReLU',
        ...     aux_channels=80,
        ...     aux_context_window=2
        ... )
        >>> input_tensor = torch.randn(1, 80, 100)  # (B, C, T_feats)
        >>> output_tensor = model(input_tensor)
        >>> output_tensor.shape
        torch.Size([1, 80, 400])  # (B, C, T_wav) where T_wav = T_feats * prod(upsample_scales)
    """

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
            aux_channels,
            aux_channels,
            kernel_size=kernel_size,
            bias=False,
        )
        self.upsample = UpsampleNetwork(
            upsample_scales=upsample_scales,
            nonlinear_activation=nonlinear_activation,
            nonlinear_activation_params=nonlinear_activation_params,
            interpolate_mode=interpolate_mode,
            freq_axis_kernel_size=freq_axis_kernel_size,
        )

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        """
            Calculate forward propagation.

        This method performs forward propagation through the convolutional
        and upsampling layers of the ConvInUpsampleNetwork. It processes the
        input tensor and produces an upsampled output tensor.

        Args:
            c (Tensor): Input tensor with shape (B, C, T_feats), where:
                - B is the batch size.
                - C is the number of channels.
                - T_feats is the number of feature frames.

        Returns:
            Tensor: Upsampled tensor with shape (B, C, T_wav), where:
                T_wav = T_feats * prod(upsample_scales), representing the
                total number of time steps after upsampling.

        Examples:
            >>> model = ConvInUpsampleNetwork(upsample_scales=[2, 2])
            >>> input_tensor = torch.randn(4, 80, 10)  # Example input
            >>> output_tensor = model(input_tensor)
            >>> output_tensor.shape
            torch.Size([4, 80, 40])  # Example output shape after upsampling

        Note:
            The upsampling is performed sequentially through a series of
            Stretch2d and Conv2d layers, followed by an optional
            nonlinear activation function.
        """
        c = self.conv_in(self.pad(c))
        return self.upsample(c)

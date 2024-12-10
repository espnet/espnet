# Adapted from https://github.com/facebookresearch/encodec by Jiatong Shi

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in https://github.com/facebookresearch/encodec/tree/main

"""Encodec SEANet-based encoder and decoder implementation."""


import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import einops
import numpy as np
import torch
from packaging.version import parse as V
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm

from espnet2.gan_codec.shared.encoder.snake_activation import Snake1d

if V(torch.__version__) >= V("2.1.0"):
    from torch.nn.utils.parametrizations import weight_norm
else:
    from torch.nn.utils import weight_norm


CONV_NORMALIZATIONS = frozenset(
    [
        "none",
        "weight_norm",
        "spectral_norm",
        "time_layer_norm",
        "layer_norm",
        "time_group_norm",
    ]
)


class ConvLayerNorm(nn.LayerNorm):
    """
    Convolution-friendly LayerNorm that moves channels to the last dimensions
    before running the normalization and moves them back to the original
    position right after.

    This layer is particularly useful for convolutional neural networks
    where normalization needs to be applied along the channel dimension.

    Args:
        normalized_shape (Union[int, List[int], torch.Size]): Shape of the
            input tensor that will be normalized. This can be an integer or
            a list/tuple of integers specifying the size of each dimension
            to normalize over.
        **kwargs: Additional keyword arguments to pass to the parent
            LayerNorm class.

    Returns:
        torch.Tensor: The normalized output tensor with the same shape as
        the input tensor.

    Examples:
        >>> import torch
        >>> layer_norm = ConvLayerNorm(normalized_shape=16)
        >>> input_tensor = torch.randn(8, 16, 50)  # (batch_size, channels, time)
        >>> output_tensor = layer_norm(input_tensor)
        >>> output_tensor.shape
        torch.Size([8, 16, 50])

    Note:
        This implementation is optimized for use in convolutional layers
        where the input tensor shape is typically (batch_size, channels,
        time).
    """

    def __init__(self, normalized_shape: Union[int, List[int], torch.Size], **kwargs):
        super().__init__(normalized_shape, **kwargs)

    def forward(self, x):
        """
            Applies the convolutional layer normalization to the input tensor.

        This method rearranges the input tensor to ensure that the channels
        are in the last dimension before applying the LayerNorm. After the
        normalization, it rearranges the tensor back to its original shape.

        Args:
            x (torch.Tensor): The input tensor with shape (B, C, T), where
                B is the batch size, C is the number of channels, and T is
                the length of the sequence.

        Returns:
            torch.Tensor: The normalized output tensor with the same shape as
            the input tensor.

        Examples:
            >>> layer_norm = ConvLayerNorm(normalized_shape=64)
            >>> input_tensor = torch.randn(32, 64, 100)  # (B, C, T)
            >>> output_tensor = layer_norm(input_tensor)
            >>> output_tensor.shape
            torch.Size([32, 64, 100])
        """
        x = einops.rearrange(x, "b ... t -> b t ...")
        x = super().forward(x)
        x = einops.rearrange(x, "b t ... -> b ... t")
        return


def apply_parametrization_norm(module: nn.Module, norm: str = "none") -> nn.Module:
    """
    Applies a specified normalization technique to a given module.

    This function checks the normalization method specified in the `norm` argument
    and applies the corresponding normalization to the input `module`. If the
    normalization type is not recognized, the original module is returned unchanged.

    Args:
        module (nn.Module): The neural network module to which the normalization
            will be applied.
        norm (str, optional): The type of normalization to apply. It can be one
            of the following: 'none', 'weight_norm', or 'spectral_norm'.
            Defaults to 'none'.

    Returns:
        nn.Module: The module with the specified normalization applied.

    Raises:
        AssertionError: If the specified normalization type is not in the
            predefined set of allowed normalizations.

    Examples:
        >>> import torch.nn as nn
        >>> conv_layer = nn.Conv2d(3, 64, kernel_size=3)
        >>> normalized_layer = apply_parametrization_norm(conv_layer, 'weight_norm')
        >>> assert isinstance(normalized_layer, nn.utils.weight_norm)
    """
    assert norm in CONV_NORMALIZATIONS
    if norm == "weight_norm":
        return weight_norm(module)
    elif norm == "spectral_norm":
        return spectral_norm(module)
    else:
        # We already check was in CONV_NORMALIZATION, so any other choice
        # doesn't need reparametrization.
        return module


def get_norm_module(
    module: nn.Module, causal: bool = False, norm: str = "none", **norm_kwargs
) -> nn.Module:
    """
        Return the proper normalization module. If causal is True, this will ensure the
    returned module is causal, or return an error if the normalization doesn't support
    causal evaluation.

    Args:
        module (nn.Module): The module to which normalization will be applied.
        causal (bool): Indicates whether to enforce causal normalization.
        norm (str): The type of normalization to apply. Options include:
            - "none": No normalization.
            - "weight_norm": Weight normalization.
            - "spectral_norm": Spectral normalization.
            - "time_layer_norm": Time layer normalization.
            - "layer_norm": Layer normalization.
            - "time_group_norm": Time group normalization.
        **norm_kwargs: Additional keyword arguments specific to the normalization
            method.

    Returns:
        nn.Module: The module with the specified normalization applied.

    Raises:
        AssertionError: If the specified normalization type is not in
            `CONV_NORMALIZATIONS`.
        ValueError: If causal is True and the specified normalization type does not
            support causal evaluation (e.g., GroupNorm).

    Examples:
        >>> conv_module = nn.Conv1d(16, 33, kernel_size=3)
        >>> norm_module = get_norm_module(conv_module, causal=False, norm="layer_norm")
        >>> assert isinstance(norm_module, ConvLayerNorm)

        >>> norm_module = get_norm_module(conv_module, causal=True, norm="layer_norm")
        >>> assert isinstance(norm_module, ConvLayerNorm)

        >>> norm_module = get_norm_module(conv_module, causal=True, norm="time_group_norm")
        ValueError: GroupNorm doesn't support causal evaluation.
    """
    assert norm in CONV_NORMALIZATIONS
    if norm == "layer_norm":
        assert isinstance(module, nn.modules.conv._ConvNd)
        return ConvLayerNorm(module.out_channels, **norm_kwargs)
    elif norm == "time_group_norm":
        if causal:
            raise ValueError("GroupNorm doesn't support causal evaluation.")
        assert isinstance(module, nn.modules.conv._ConvNd)
        num_groups = norm_kwargs.pop("num_groups", 1)
        return nn.GroupNorm(num_groups, module.out_channels, **norm_kwargs)
    else:
        return nn.Identity()


def get_extra_padding_for_conv1d(
    x: torch.Tensor, kernel_size: int, stride: int, padding_total: int = 0
) -> int:
    """
        Calculate the extra padding required for a 1D convolution operation to ensure
    that the last window is fully utilized. This is important for reconstructing an
    output of the same length, as some time steps may be removed even with padding.

    The function computes how much additional padding is needed at the end of the
    input tensor to ensure that the output tensor retains the correct size.

    For instance, with total padding = 4, kernel size = 4, stride = 2:
            0 0 1 2 3 4 5 0 0 # (0s are padding)
            1   2   3         # (out-frames of a convolution, last 0 is never used)
            0 0 1 2 3 4 5 0   # (out-tr.conv., but pos.5 will get removed as padding)
                1 2 3 4       # once you removed padding, we are missing one time step !

    Args:
        x (torch.Tensor): The input tensor of shape (batch_size, channels, length).
        kernel_size (int): The size of the convolutional kernel.
        stride (int): The stride of the convolution.
        padding_total (int, optional): The total padding applied to the input.
            Defaults to 0.

    Returns:
        int: The amount of extra padding needed at the end of the input tensor.

    Examples:
        >>> import torch
        >>> x = torch.randn(1, 1, 6)  # (batch_size=1, channels=1, length=6)
        >>> kernel_size = 4
        >>> stride = 2
        >>> padding_total = 4
        >>> extra_padding = get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)
        >>> print(extra_padding)  # Outputs the extra padding required

    Note:
        This function is crucial for ensuring that all time steps are processed
        during convolution, especially in scenarios where the input length is
        not perfectly divisible by the stride.
    """
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length


def pad1d(
    x: torch.Tensor, paddings: Tuple[int, int], mode: str = "zero", value: float = 0.0
):
    """
    Apply padding to a 1D tensor with support for reflection and other modes.

    This function serves as a lightweight wrapper around `torch.nn.functional.pad`
    to facilitate the handling of reflection padding when the input tensor is small.
    If the input length is less than or equal to the maximum padding size, extra
    zero padding is added to ensure the reflection operation can be performed
    without errors.

    Args:
        x (torch.Tensor): The input tensor of shape (..., length).
        paddings (Tuple[int, int]): A tuple specifying the padding sizes for the
            left and right sides, respectively.
        mode (str, optional): The padding mode to use. Supported modes are:
            'zero', 'reflect', 'replicate', and 'circular'. Default is 'zero'.
        value (float, optional): The value to use for padding when `mode` is 'zero'.
            Default is 0.0.

    Returns:
        torch.Tensor: The padded tensor.

    Raises:
        AssertionError: If the padding values are negative.

    Examples:
        >>> import torch
        >>> x = torch.tensor([1, 2, 3, 4])
        >>> padded = pad1d(x, (2, 2), mode='zero')
        >>> print(padded)
        tensor([0, 0, 1, 2, 3, 4, 0, 0])

        >>> padded_reflect = pad1d(x, (2, 2), mode='reflect')
        >>> print(padded_reflect)
        tensor([3, 4, 3, 2, 1, 2, 3, 4])
    """
    length = x.shape[-1]
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    if mode == "reflect":
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))
        padded = F.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    else:
        return F.pad(x, paddings, mode, value)


class NormConv1d(nn.Module):
    """
    Wrapper around Conv1d and normalization applied to this conv
    to provide a uniform interface across normalization approaches.

    This class applies a 1D convolution followed by a specified
    normalization method, allowing flexibility in the choice of
    normalization technique while maintaining a consistent interface.

    Attributes:
        conv (nn.Module): The convolutional layer with optional
            parameterization normalization.
        norm (nn.Module): The normalization layer applied after
            the convolution.
        norm_type (str): The type of normalization used.

    Args:
        *args: Variable length argument list for the Conv1d layer.
        causal (bool, optional): Whether to use causal convolution.
            Defaults to False.
        norm (str, optional): The normalization method to apply.
            Options include "none", "weight_norm", "spectral_norm",
            "time_layer_norm", "layer_norm", "time_group_norm".
            Defaults to "none".
        norm_kwargs (Dict[str, Any], optional): Additional keyword
            arguments for the normalization layer. Defaults to an
            empty dictionary.

    Examples:
        >>> norm_conv = NormConv1d(in_channels=16, out_channels=33,
        ...                         kernel_size=3, causal=True,
        ...                         norm="layer_norm")
        >>> input_tensor = torch.randn(10, 16, 50)  # (batch_size, channels, length)
        >>> output_tensor = norm_conv(input_tensor)
        >>> output_tensor.shape
        torch.Size([10, 33, 48])  # Output shape after convolution
    """

    def __init__(
        self,
        *args,
        causal: bool = False,
        norm: str = "none",
        norm_kwargs: Dict[str, Any] = {},
        **kwargs,
    ):
        super().__init__()
        self.conv = apply_parametrization_norm(nn.Conv1d(*args, **kwargs), norm)
        self.norm = get_norm_module(self.conv, causal, norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x):
        """
            Applies the convolution followed by the normalization to the input.

        This method takes an input tensor `x`, applies a 1D convolution using the
        defined `Conv1d` layer, and subsequently normalizes the result using the
        specified normalization technique. It provides a unified interface for
        various normalization approaches.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, in_channels,
                              sequence_length) to be processed.

        Returns:
            torch.Tensor: The output tensor after applying the convolution and
                          normalization, with the same shape as the input tensor.

        Examples:
            >>> conv_layer = NormConv1d(in_channels=1, out_channels=2, kernel_size=3)
            >>> input_tensor = torch.randn(4, 1, 10)  # (batch_size=4, channels=1, length=10)
            >>> output_tensor = conv_layer(input_tensor)
            >>> output_tensor.shape
            torch.Size([4, 2, 8])  # The output shape will vary based on kernel size and stride

        Note:
            Ensure that the input tensor is of the correct shape and type. The
            method expects a 3D tensor and will raise an error if the input does
            not conform to this requirement.
        """
        x = self.conv(x)
        x = self.norm(x)
        return x


class SConv1d(nn.Module):
    """
    Conv1d with built-in handling of asymmetric or causal padding and
    normalization.

    This module is designed to facilitate convolution operations with
    options for different padding strategies and normalization methods.
    It automatically computes necessary padding based on the specified
    convolution parameters, ensuring that the output shape aligns with
    the expected dimensions, particularly in causal scenarios.

    Attributes:
        conv (NormConv1d): A normalized 1D convolution layer.
        causal (bool): Indicates if causal padding should be used.
        pad_mode (str): Specifies the padding mode ('reflect', 'zero', etc.).

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Default is 1.
        dilation (int, optional): Dilation of the convolution. Default is 1.
        groups (int, optional): Number of blocked connections. Default is 1.
        bias (bool, optional): If True, adds a learnable bias to the output.
            Default is True.
        causal (bool, optional): If True, applies causal padding. Default is False.
        norm (str, optional): Normalization method. Default is 'none'.
        norm_kwargs (Dict[str, Any], optional): Additional arguments for normalization.
        pad_mode (str, optional): Padding mode for the convolutions.
            Default is 'reflect'.

    Raises:
        ValueError: If both stride and dilation are greater than 1.

    Examples:
        >>> conv = SConv1d(in_channels=16, out_channels=33, kernel_size=3)
        >>> input_tensor = torch.randn(20, 16, 50)  # (batch_size, channels, time)
        >>> output_tensor = conv(input_tensor)
        >>> output_tensor.shape
        torch.Size([20, 33, 48])  # Output shape with default parameters

    Note:
        The effective kernel size is calculated by taking dilation into account.
        For causal convolutions, left padding is applied to maintain causality.

    Todo:
        Consider adding support for other normalization techniques in the future.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        causal: bool = False,
        norm: str = "none",
        norm_kwargs: Dict[str, Any] = {},
        pad_mode: str = "reflect",
    ):
        super().__init__()
        # warn user on unusual setup between dilation and stride
        if stride > 1 and dilation > 1:
            logging.warning(
                "SConv1d has been initialized with stride > 1 and dilation > 1"
                f" (kernel_size={kernel_size} stride={stride}, dilation={dilation})."
            )
        self.conv = NormConv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            causal=causal,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )
        self.causal = causal
        self.pad_mode = pad_mode

    def forward(self, x):
        """
                Forward pass for the SConv1d module.

        This method processes the input tensor through a 1D convolutional layer,
        applies necessary padding, and normalizes the output if specified. It
        handles both causal and non-causal convolutions and manages the
        padding appropriately based on the convolution parameters.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, T), where B is the batch
                              size, C is the number of channels, and T is the
                              sequence length.

        Returns:
            torch.Tensor: The output tensor after applying the convolution and
                          normalization, with the same shape as the input tensor
                          if padding is correctly applied.

        Examples:
            >>> model = SConv1d(in_channels=1, out_channels=2, kernel_size=3)
            >>> input_tensor = torch.randn(5, 1, 10)  # Batch size of 5, 1 channel, length 10
            >>> output_tensor = model(input_tensor)
            >>> output_tensor.shape
            torch.Size([5, 2, 10])  # Output shape after convolution and normalization
        """
        B, C, T = x.shape
        kernel_size = self.conv.conv.kernel_size[0]
        stride = self.conv.conv.stride[0]
        dilation = self.conv.conv.dilation[0]
        kernel_size = (
            kernel_size - 1
        ) * dilation + 1  # effective kernel size with dilations
        padding_total = kernel_size - stride
        extra_padding = get_extra_padding_for_conv1d(
            x, kernel_size, stride, padding_total
        )
        if self.causal:
            # Left padding for causal
            x = pad1d(x, (padding_total, extra_padding), mode=self.pad_mode)
        else:
            # Asymmetric padding required for odd strides
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            x = pad1d(
                x, (padding_left, padding_right + extra_padding), mode=self.pad_mode
            )
        return self.conv(x)


class SLSTM(nn.Module):
    """
        SLSTM is a custom Long Short-Term Memory (LSTM) module designed to handle
    inputs with a convolutional layout. It abstracts the complexities of hidden
    state management and input data arrangement, providing a simplified interface
    for sequential data processing.

    Attributes:
        skip (bool): If True, adds the input to the output of the LSTM for a skip
            connection. Defaults to True.
        lstm (nn.LSTM): The LSTM layer used for processing the input data.

    Args:
        dimension (int): The number of expected features in the input (also the
            number of output features).
        num_layers (int, optional): The number of recurrent layers. Defaults to 2.
        skip (bool, optional): Whether to use a skip connection by adding the
            input to the output. Defaults to True.

    Returns:
        torch.Tensor: The output tensor, with the same shape as the input tensor
        but with the features processed by the LSTM.

    Examples:
        >>> model = SLSTM(dimension=128)
        >>> input_tensor = torch.randn(10, 32, 128)  # (batch_size, seq_len, features)
        >>> output_tensor = model(input_tensor)
        >>> print(output_tensor.shape)  # Output shape: (10, 32, 128)

    Note:
        The input tensor is expected to be in the shape of (batch_size, seq_len,
        features) and is permuted to (seq_len, batch_size, features) before being
        passed to the LSTM layer.

    Todo:
        - Consider adding support for customizable hidden states if required.
    """

    def __init__(self, dimension: int, num_layers: int = 2, skip: bool = True):
        super().__init__()
        self.skip = skip
        self.lstm = nn.LSTM(dimension, dimension, num_layers)

    def forward(self, x):
        """
            Applies the LSTM to the input tensor and optionally adds the input
        tensor to the output for skip connections.

        The input tensor is expected to be in a convolutional layout with the
        shape (batch_size, channels, sequence_length). The output will have
        the same shape as the input tensor.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, channels,
            sequence_length).

        Returns:
            torch.Tensor: Output tensor after applying LSTM and skip connection
            if enabled.

        Examples:
            >>> slstm = SLSTM(dimension=128)
            >>> input_tensor = torch.randn(32, 128, 100)  # (batch_size, channels, seq_len)
            >>> output_tensor = slstm(input_tensor)
            >>> print(output_tensor.shape)  # Should be (32, 128, 100)

        Note:
            The input tensor is permuted to (sequence_length, batch_size,
            channels) before being passed to the LSTM.
        """
        x = x.permute(2, 0, 1)
        y, _ = self.lstm(x)
        if self.skip:
            y = y + x
        y = y.permute(1, 2, 0)
        return y


class SEANetResnetBlock(nn.Module):
    """
    Residual block from SEANet model.

    This class implements a residual block for the SEANet architecture, which
    consists of a series of convolutional layers, activation functions, and
    normalization layers. It is designed to facilitate efficient training of
    deep networks while preserving important features through skip connections.

    Args:
        dim (int): Dimension of the input/output.
        kernel_sizes (list): List of kernel sizes for the convolutions.
        dilations (list): List of dilations for the convolutions.
        activation (str): Activation function.
        activation_params (dict): Parameters to provide to the activation
            function.
        norm (str): Normalization method.
        norm_params (dict): Parameters to provide to the underlying
            normalization used along with the convolution.
        causal (bool): Whether to use fully causal convolution.
        pad_mode (str): Padding mode for the convolutions.
        compress (int): Reduced dimensionality in residual branches
            (from Demucs v3).
        true_skip (bool): Whether to use true skip connection or a simple
            convolution as the skip connection.

    Examples:
        >>> block = SEANetResnetBlock(dim=128, kernel_sizes=[3, 1],
        ...                            dilations=[1, 1], activation='ELU',
        ...                            activation_params={'alpha': 1.0},
        ...                            norm='weight_norm',
        ...                            norm_params={}, causal=False,
        ...                            pad_mode='reflect', compress=2,
        ...                            true_skip=True)
        >>> x = torch.randn(10, 128, 50)  # Batch of 10, 128 channels, 50 time steps
        >>> output = block(x)
        >>> output.shape
        torch.Size([10, 128, 50])

    Note:
        The block utilizes skip connections to improve gradient flow and
        reduce the risk of vanishing gradients in deeper networks.

    Raises:
        AssertionError: If the number of kernel sizes does not match the
        number of dilations.
    """

    def __init__(
        self,
        dim: int,
        kernel_sizes: List[int] = [3, 1],
        dilations: List[int] = [1, 1],
        activation: str = "ELU",
        activation_params: dict = {"alpha": 1.0},
        norm: str = "weight_norm",
        norm_params: Dict[str, Any] = {},
        causal: bool = False,
        pad_mode: str = "reflect",
        compress: int = 2,
        true_skip: bool = True,
    ):
        super().__init__()
        assert len(kernel_sizes) == len(
            dilations
        ), "Number of kernel sizes should match number of dilations"

        if activation == "Snake":
            act = Snake1d
        else:
            act = getattr(nn, activation)
        hidden = dim // compress
        block = []
        for i, (kernel_size, dilation) in enumerate(zip(kernel_sizes, dilations)):
            in_chs = dim if i == 0 else hidden
            out_chs = dim if i == len(kernel_sizes) - 1 else hidden
            block += [
                act(**activation_params),
                SConv1d(
                    in_chs,
                    out_chs,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    norm=norm,
                    norm_kwargs=norm_params,
                    causal=causal,
                    pad_mode=pad_mode,
                ),
            ]
        self.block = nn.Sequential(*block)
        self.shortcut: nn.Module
        if true_skip:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = SConv1d(
                dim,
                dim,
                kernel_size=1,
                norm=norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            )

    def forward(self, x):
        """
            Residual block from SEANet model.

        This class implements a residual block used in the SEANet model, which
        consists of convolutional layers with normalization and activation
        functions. It allows for causal convolutions and provides options for
        different normalization methods.

        Attributes:
            block (nn.Sequential): A sequential container of activation and
                convolutional layers.
            shortcut (nn.Module): A skip connection that can be either an
                identity mapping or a convolutional layer.

        Args:
            dim (int): Dimension of the input/output.
            kernel_sizes (list): List of kernel sizes for the convolutions.
            dilations (list): List of dilations for the convolutions.
            activation (str): Activation function.
            activation_params (dict): Parameters to provide to the activation
                function.
            norm (str): Normalization method.
            norm_params (dict): Parameters to provide to the underlying
                normalization used along with the convolution.
            causal (bool): Whether to use fully causal convolution.
            pad_mode (str): Padding mode for the convolutions.
            compress (int): Reduced dimensionality in residual branches
                (from Demucs v3).
            true_skip (bool): Whether to use true skip connection or a simple
                convolution as the skip connection.

        Examples:
            >>> block = SEANetResnetBlock(dim=128)
            >>> input_tensor = torch.randn(1, 128, 64)  # (batch, channels, time)
            >>> output_tensor = block(input_tensor)
            >>> output_tensor.shape
            torch.Size([1, 128, 64])

        Raises:
            AssertionError: If the number of kernel sizes does not match the
                number of dilations.

        Note:
            This block is designed to work with various normalization methods,
            including weight normalization and layer normalization. It can be
            easily adapted to different activation functions.
        """
        return self.shortcut(x) + self.block(x)


class SEANetEncoder(nn.Module):
    """
        SEANet encoder.

    This class implements the SEANet encoder, which is a neural network
    architecture designed for audio processing tasks. The encoder utilizes
    convolutional layers, residual blocks, and optional LSTM layers to
    extract features from audio input.

    Attributes:
        channels (int): Number of audio channels (default is 1).
        dimension (int): Dimension of the intermediate representation
            (default is 128).
        n_filters (int): Base width for the model (default is 32).
        n_residual_layers (int): Number of residual layers (default is 1).
        ratios (List[int]): Downsampling ratios (default is [8, 5, 4, 2]).
        activation (str): Activation function (default is "ELU").
        activation_params (dict): Parameters for the activation function
            (default is {"alpha": 1.0}).
        norm (str): Normalization method (default is "weight_norm").
        norm_params (dict): Parameters for the underlying normalization
            used with the convolution (default is an empty dictionary).
        kernel_size (int): Kernel size for the initial convolution
            (default is 7).
        last_kernel_size (int): Kernel size for the last convolution
            (default is 7).
        residual_kernel_size (int): Kernel size for the residual layers
            (default is 3).
        dilation_base (int): Base value for increasing dilation with each
            layer (default is 2).
        causal (bool): Whether to use fully causal convolution (default is
            False).
        pad_mode (str): Padding mode for convolutions (default is "reflect").
        true_skip (bool): Whether to use true skip connections or a
            simple convolution as the skip connection in the residual
            blocks (default is False).
        compress (int): Reduced dimensionality in residual branches
            (default is 2).
        lstm (int): Number of LSTM layers at the end of the encoder
            (default is 2).

    Args:
        channels (int): Audio channels.
        dimension (int): Intermediate representation dimension.
        n_filters (int): Base width for the model.
        n_residual_layers (int): Number of residual layers.
        ratios (Sequence[int]): Kernel size and stride ratios. The encoder
            uses downsampling ratios instead of upsampling ratios, hence
            it will use the ratios in the reverse order to the ones
            specified here that must match the decoder order.
        activation (str): Activation function.
        activation_params (dict): Parameters to provide to the activation
            function.
        norm (str): Normalization method.
        norm_params (dict): Parameters to provide to the underlying
            normalization used along with the convolution.
        kernel_size (int): Kernel size for the initial convolution.
        last_kernel_size (int): Kernel size for the last convolution.
        residual_kernel_size (int): Kernel size for the residual layers.
        dilation_base (int): How much to increase the dilation with each
            layer.
        causal (bool): Whether to use fully causal convolution.
        pad_mode (str): Padding mode for the convolutions.
        true_skip (bool): Whether to use true skip connection or a simple
            (streamable) convolution as the skip connection in the
            residual network blocks.
        compress (int): Reduced dimensionality in residual branches
            (from Demucs v3).
        lstm (int): Number of LSTM layers at the end of the encoder.

    Examples:
        >>> encoder = SEANetEncoder(channels=1, dimension=128)
        >>> input_tensor = torch.randn(1, 1, 16000)  # Batch size of 1, 1 channel, 16000 samples
        >>> output = encoder(input_tensor)
        >>> print(output.shape)
        torch.Size([1, 128, 2000])  # Example output shape

    Note:
        This encoder is part of a larger audio processing framework and
        is intended for use in GAN-based audio synthesis tasks.
    """

    def __init__(
        self,
        channels: int = 1,
        dimension: int = 128,
        n_filters: int = 32,
        n_residual_layers: int = 1,
        ratios: List[int] = [8, 5, 4, 2],
        activation: str = "ELU",
        activation_params: dict = {"alpha": 1.0},
        norm: str = "weight_norm",
        norm_params: Dict[str, Any] = {},
        kernel_size: int = 7,
        last_kernel_size: int = 7,
        residual_kernel_size: int = 3,
        dilation_base: int = 2,
        causal: bool = False,
        pad_mode: str = "reflect",
        true_skip: bool = False,
        compress: int = 2,
        lstm: int = 2,
    ):
        super().__init__()
        self.channels = channels
        self.dimension = dimension
        self.n_filters = n_filters
        self.ratios = list(reversed(ratios))
        del ratios
        self.n_residual_layers = n_residual_layers
        self.hop_length = np.prod(self.ratios)

        if activation == "Snake":
            act = Snake1d
        else:
            act = getattr(nn, activation)
        mult = 1
        model: List[nn.Module] = [
            SConv1d(
                channels,
                mult * n_filters,
                kernel_size,
                norm=norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            )
        ]
        # Downsample to raw audio scale
        for i, ratio in enumerate(self.ratios):
            # Add residual layers
            for j in range(n_residual_layers):
                model += [
                    SEANetResnetBlock(
                        mult * n_filters,
                        kernel_sizes=[residual_kernel_size, 1],
                        dilations=[dilation_base**j, 1],
                        norm=norm,
                        norm_params=norm_params,
                        activation=activation,
                        activation_params=activation_params,
                        causal=causal,
                        pad_mode=pad_mode,
                        compress=compress,
                        true_skip=true_skip,
                    )
                ]

            # Add downsampling layers
            model += [
                act(**activation_params),
                SConv1d(
                    mult * n_filters,
                    mult * n_filters * 2,
                    kernel_size=ratio * 2,
                    stride=ratio,
                    norm=norm,
                    norm_kwargs=norm_params,
                    causal=causal,
                    pad_mode=pad_mode,
                ),
            ]
            mult *= 2

        if lstm:
            model += [SLSTM(mult * n_filters, num_layers=lstm)]

        model += [
            act(**activation_params),
            SConv1d(
                mult * n_filters,
                dimension,
                last_kernel_size,
                norm=norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            ),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        """
            SEANet encoder.

        This class implements the SEANet encoder architecture for audio processing.
        It includes multiple convolutional layers, normalization, activation functions,
        and optional LSTM layers to produce an intermediate representation of audio
        signals.

        Attributes:
            channels (int): Number of audio channels (default: 1).
            dimension (int): Intermediate representation dimension (default: 128).
            n_filters (int): Base width for the model (default: 32).
            n_residual_layers (int): Number of residual layers (default: 1).
            ratios (List[int]): Kernel size and stride ratios, must match the decoder
                order (default: [8, 5, 4, 2]).
            activation (str): Activation function to use (default: "ELU").
            activation_params (dict): Parameters for the activation function
                (default: {"alpha": 1.0}).
            norm (str): Normalization method to use (default: "weight_norm").
            norm_params (dict): Parameters for the normalization method
                (default: {}).
            kernel_size (int): Kernel size for the initial convolution (default: 7).
            last_kernel_size (int): Kernel size for the final convolution (default: 7).
            residual_kernel_size (int): Kernel size for the residual layers (default: 3).
            dilation_base (int): How much to increase the dilation with each layer
                (default: 2).
            causal (bool): Whether to use fully causal convolution (default: False).
            pad_mode (str): Padding mode for the convolutions (default: "reflect").
            true_skip (bool): Whether to use true skip connection or a simple
                convolution as the skip connection in residual blocks (default: False).
            compress (int): Reduced dimensionality in residual branches (from Demucs v3,
                default: 2).
            lstm (int): Number of LSTM layers at the end of the encoder (default: 2).

        Args:
            channels (int): Audio channels.
            dimension (int): Intermediate representation dimension.
            n_filters (int): Base width for the model.
            n_residual_layers (int): Number of residual layers.
            ratios (Sequence[int]): Kernel size and stride ratios.
            activation (str): Activation function.
            activation_params (dict): Parameters for the activation function.
            norm (str): Normalization method.
            norm_params (dict): Parameters for the normalization method.
            kernel_size (int): Kernel size for the initial convolution.
            last_kernel_size (int): Kernel size for the final convolution.
            residual_kernel_size (int): Kernel size for the residual layers.
            dilation_base (int): Dilation increment for each layer.
            causal (bool): Use fully causal convolution.
            pad_mode (str): Padding mode for convolutions.
            true_skip (bool): Use true skip connection.
            compress (int): Reduced dimensionality in residual branches.
            lstm (int): Number of LSTM layers at the end of the encoder.

        Examples:
            >>> encoder = SEANetEncoder(channels=1, dimension=128)
            >>> audio_input = torch.randn(1, 1, 16000)  # (batch_size, channels, length)
            >>> output = encoder(audio_input)
            >>> print(output.shape)  # Should be (1, 128, length after processing)

        Note:
            The `ratios` attribute defines the downsampling factors used in the
            encoder, which should be specified in reverse order compared to the
            decoder.

        Raises:
            AssertionError: If the number of kernel sizes does not match the number
                of dilations.
        """
        return self.model(x)

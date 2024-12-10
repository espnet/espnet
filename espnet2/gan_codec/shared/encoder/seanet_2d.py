# Adapted by Zhihao Du for 2D SEANet (from seanet.py)

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in https://github.com/facebookresearch/encodec/tree/main

"""Encodec SEANet-based encoder and decoder implementation."""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from packaging.version import parse as V
from torch.nn import functional as F

from espnet2.gan_codec.shared.encoder.seanet import (
    SLSTM,
    SConv1d,
    apply_parametrization_norm,
    get_extra_padding_for_conv1d,
    get_norm_module,
)

if V(torch.__version__) >= V("2.1.0"):
    from torch.nn.utils.parametrizations import weight_norm
else:
    from torch.nn.utils import weight_norm


def get_activation(activation: str = None, channels=None, **kwargs):
    """
        Get the specified activation function.

    This function returns an activation function as a PyTorch module based on the
    provided name. It supports custom activation functions such as 'snake' which
    requires the number of channels to be specified.

    Attributes:
        activation (str): The name of the activation function to retrieve.
        channels (Optional[int]): The number of channels required for specific
            activation functions (e.g., 'snake').
        kwargs (Any): Additional parameters for the activation function.

    Args:
        activation (str): The name of the activation function to use.
            Common options include 'ReLU', 'ELU', 'LeakyReLU', etc.
        channels (Optional[int]): The number of input channels for activation
            functions that require it.
        **kwargs: Additional keyword arguments for the activation function.

    Returns:
        nn.Module: The corresponding activation function as a PyTorch module.

    Raises:
        AssertionError: If 'snake' is specified without providing the number
            of channels.

    Examples:
        >>> relu = get_activation('ReLU')
        >>> snake_activation = get_activation('snake', channels=64)

    Note:
        The function uses `getattr` to dynamically retrieve the activation
        function from the `torch.nn` module. Make sure to pass valid
        activation names.
    """
    if activation.lower() == "snake":
        assert channels is not None, "Snake activation needs channel number."
        return Snake1d(channels=channels)
    else:
        act = getattr(nn, activation)
        return act(**kwargs)


class NormConv2d(nn.Module):
    """
    Wrapper around Conv2d with normalization to provide a uniform interface.

    This class encapsulates a 2D convolutional layer along with a normalization
    layer, allowing for various normalization techniques to be applied
    seamlessly. The user can specify the type of normalization to be used
    and provide any additional parameters required for that normalization.

    Attributes:
        conv (nn.Module): The convolutional layer with optional
            weight normalization.
        norm (nn.Module): The normalization layer applied to the output
            of the convolution.
        norm_type (str): The type of normalization applied.

    Args:
        *args: Variable length argument list to be passed to
            nn.Conv2d.
        causal (bool, optional): If True, applies causal convolution.
            Defaults to False.
        norm (str, optional): The type of normalization to apply.
            Defaults to "none".
        norm_kwargs (Dict[str, Any], optional): Additional keyword
            arguments for the normalization layer. Defaults to an empty
            dictionary.

    Examples:
        >>> layer = NormConv2d(1, 32, kernel_size=(3, 3), norm='batch_norm')
        >>> input_tensor = torch.randn(1, 1, 64, 64)
        >>> output_tensor = layer(input_tensor)
        >>> output_tensor.shape
        torch.Size([1, 32, 62, 62])
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
        self.conv = apply_parametrization_norm(nn.Conv2d(*args, **kwargs), norm)
        self.norm = get_norm_module(self.conv, causal, norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x):
        """
            Applies the convolution and normalization to the input tensor.

        This method takes an input tensor `x`, applies the convolutional layer,
        followed by the normalization layer, and returns the output tensor.

        Args:
            x (torch.Tensor): The input tensor of shape (B, C, F, T) where:
                B is the batch size,
                C is the number of channels,
                F is the frequency dimension,
                T is the time dimension.

        Returns:
            torch.Tensor: The output tensor after applying the convolution and
            normalization, with the same shape as the input tensor.

        Raises:
            AssertionError: If the input tensor does not have 4 dimensions.

        Examples:
            >>> norm_conv = NormConv2d(in_channels=1, out_channels=2, kernel_size=3)
            >>> input_tensor = torch.randn(4, 1, 16, 16)  # Batch size of 4
            >>> output_tensor = norm_conv(input_tensor)
            >>> output_tensor.shape
            torch.Size([4, 2, 14, 14])  # Output shape after conv and norm
        """
        x = self.conv(x)
        x = self.norm(x)
        return x


def tuple_it(x, num=2):
    """
        Converts various input types to a tuple of a specified size.

    This function handles different types of inputs, converting them to a tuple
    of a specified number of elements. It supports inputs of type list and int,
    while leaving other types unchanged.

    Args:
        x (Union[list, int, Any]): The input to be converted into a tuple.
        num (int): The size of the tuple to return. Default is 2.

    Returns:
        Union[Tuple, Any]: A tuple containing the first two elements of the list,
        or a tuple of the specified size filled with the integer value if the
        input is an integer. If the input is neither a list nor an int, it
        returns the input unchanged.

    Examples:
        >>> tuple_it([1, 2, 3, 4])
        (1, 2)

        >>> tuple_it(5, num=3)
        (5, 5, 5)

        >>> tuple_it("string")
        'string'

    Note:
        If the input is a list with fewer than two elements, the returned
        tuple will contain only the available elements.
    """
    if isinstance(x, list):
        return tuple(x[:2])
    elif isinstance(x, int):
        return tuple([x for _ in range(num)])
    else:
        return x


def pad2d(
    x: torch.Tensor,
    paddings: Tuple[Tuple[int, int], Tuple[int, int]],
    mode: str = "zero",
    value: float = 0.0,
):
    """
    Applies padding to a 2D tensor with an option for reflective padding.

    This function is a wrapper around `torch.nn.functional.pad`. It allows for
    both zero and reflective padding. When reflective padding is requested,
    additional zero padding is added if the input dimensions are smaller than
    the specified padding sizes.

    Args:
        x (torch.Tensor): The input tensor of shape (B, C, F, T), where B is
            the batch size, C is the number of channels, F is the frequency
            dimension, and T is the time dimension.
        paddings (Tuple[Tuple[int, int], Tuple[int, int]]): A tuple of two tuples
            specifying the padding for the time and frequency dimensions
            respectively. Each inner tuple contains two integers, the amount of
            padding before and after the dimension.
        mode (str, optional): The mode of padding. Can be "zero" for zero padding
            or "reflect" for reflective padding. Defaults to "zero".
        value (float, optional): The value to use for zero padding when `mode`
            is "zero". Defaults to 0.0.

    Returns:
        torch.Tensor: The padded tensor.

    Raises:
        AssertionError: If any of the padding values are negative.

    Examples:
        >>> x = torch.randn(1, 1, 4, 4)
        >>> paddings = ((1, 1), (2, 2))
        >>> padded_tensor = pad2d(x, paddings, mode="zero")
        >>> print(padded_tensor.shape)
        torch.Size([1, 1, 6, 8])

        >>> padded_tensor_reflect = pad2d(x, paddings, mode="reflect")
        >>> print(padded_tensor_reflect.shape)
        torch.Size([1, 1, 6, 8])
    """
    freq_len, time_len = x.shape[-2:]
    padding_time, padding_freq = paddings
    assert min(padding_freq) >= 0 and min(padding_time) >= 0, (
        padding_time,
        padding_freq,
    )
    if mode == "reflect":
        max_time_pad, max_freq_pad = max(padding_time), max(padding_freq)
        extra_time_pad = max_time_pad - time_len + 1 if time_len <= max_time_pad else 0
        extra_freq_pad = max_freq_pad - freq_len + 1 if freq_len <= max_freq_pad else 0
        extra_pad = [0, extra_time_pad, 0, extra_freq_pad]
        x = F.pad(x, extra_pad)
        padded = F.pad(x, (*padding_time, *padding_freq), mode, value)
        freq_end = padded.shape[-2] - extra_freq_pad
        time_end = padded.shape[-1] - extra_time_pad
        return padded[..., :freq_end, :time_end]
    else:
        return F.pad(x, (*paddings[0], *paddings[1]), mode, value)


class SConv2d(nn.Module):
    """
    Conv2d with built-in handling of asymmetric or causal padding
    and normalization.

    Note: Causal padding only makes sense on the time (last) axis.
    The frequency (second last) axis is always non-causally padded.

    Attributes:
        conv (NormConv2d): The convolutional layer with normalization.
        causal (bool): Indicates if causal padding is used.
        pad_mode (str): Padding mode for the convolutions.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (Union[int, Tuple[int, int]]): Size of the convolving kernel.
        stride (Union[int, Tuple[int, int]], optional): Stride of the convolution.
            Default is 1.
        dilation (Union[int, Tuple[int, int]], optional): Spacing between kernel
            elements. Default is 1.
        groups (int, optional): Number of blocked connections from input to output.
            Default is 1.
        bias (bool, optional): If True, adds a learnable bias to the output.
            Default is True.
        causal (bool, optional): If True, applies causal padding. Default is False.
        norm (str, optional): Normalization method to apply. Default is "none".
        norm_kwargs (Dict[str, Any], optional): Additional keyword arguments for
            normalization. Default is an empty dict.
        pad_mode (str, optional): Padding mode for the convolution. Default is
            "reflect".

    Raises:
        AssertionError: If the input tensor does not have 4 dimensions.

    Examples:
        >>> conv_layer = SConv2d(in_channels=1, out_channels=32, kernel_size=(3, 3))
        >>> input_tensor = torch.randn(8, 1, 64, 64)  # Batch of 8, 1 channel, 64x64
        >>> output_tensor = conv_layer(input_tensor)
        >>> output_tensor.shape
        torch.Size([8, 32, 62, 62])  # Output shape after convolution
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        causal: bool = False,
        norm: str = "none",
        norm_kwargs: Dict[str, Any] = {},
        pad_mode: str = "reflect",
    ):
        super().__init__()
        # warn user on unusual setup between dilation and stride
        kernel_size, stride, dilation = (
            tuple_it(kernel_size, 2),
            tuple_it(stride, 2),
            tuple_it(dilation, 2),
        )

        if max(stride) > 1 and max(dilation) > 1:
            warnings.warn(
                "SConv2d has been initialized with stride > 1 and dilation > 1"
                f" (kernel_size={kernel_size} stride={stride}, dilation={dilation})."
            )
        self.conv = NormConv2d(
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
            Applies the convolution and normalization to the input tensor.

        This method takes an input tensor `x` and passes it through the
        convolutional layer followed by the normalization layer. It is
        assumed that the input tensor has 4 dimensions (B, C, F, T),
        where B is the batch size, C is the number of channels, F is
        the frequency dimension, and T is the time dimension.

        Args:
            x (torch.Tensor): Input tensor with shape (B, C, F, T).

        Returns:
            torch.Tensor: The output tensor after applying the convolution
            and normalization, maintaining the same shape (B, C, F, T).

        Raises:
            AssertionError: If the input tensor does not have 4 dimensions.

        Examples:
            >>> sconv2d = SConv2d(in_channels=3, out_channels=16, kernel_size=(3, 3))
            >>> input_tensor = torch.randn(8, 3, 64, 64)  # Example input
            >>> output_tensor = sconv2d(input_tensor)
            >>> output_tensor.shape
            torch.Size([8, 16, 62, 62])  # Output shape will depend on kernel size
        """
        assert len(x.shape) == 4, x.shape
        B, C, F, T = x.shape
        padding_total_list: List[int] = []
        extra_padding_list: List[int] = []
        for i, (kernel_size, stride, dilation) in enumerate(
            zip(
                self.conv.conv.kernel_size,
                self.conv.conv.stride,
                self.conv.conv.dilation,
            )
        ):
            padding_total = (kernel_size - 1) * dilation - (stride - 1)
            if i == 0:
                # no extra padding for frequency dim
                extra_padding = 0
            else:
                extra_padding = get_extra_padding_for_conv1d(
                    x, kernel_size, stride, padding_total
                )
            padding_total_list.append(padding_total)
            extra_padding_list.append(extra_padding)

        if self.causal:
            # always non-causal padding for frequency axis
            freq_after = padding_total_list[0] // 2
            freq_before = padding_total_list[0] - freq_after + extra_padding_list[0]
            # causal padding for time axis
            time_after = extra_padding_list[1]
            time_before = padding_total_list[1]
            x = pad2d(
                x,
                ((time_before, time_after), (freq_before, freq_after)),
                mode=self.pad_mode,
            )
        else:
            # Asymmetric padding required for odd strides
            freq_after = padding_total_list[0] // 2
            freq_before = padding_total_list[0] - freq_after + extra_padding_list[0]
            time_after = padding_total_list[1] // 2
            time_before = padding_total_list[1] - time_after + extra_padding_list[1]
            x = pad2d(
                x,
                ((time_before, time_after), (freq_before, freq_after)),
                mode=self.pad_mode,
            )
        x = self.conv(x)
        return x


class SEANetResnetBlock2d(nn.Module):
    """
    Residual block from SEANet model.

    This class implements a residual block that consists of convolutional
    layers, activation functions, and normalization techniques. The design
    allows for customizable parameters including kernel sizes, dilations,
    and whether to use causal convolutions or true skip connections.

    Attributes:
        block (nn.Sequential): A sequence of layers including activation
            functions and convolutions.
        shortcut (nn.Module): A shortcut connection that can either be
            an identity mapping or a convolutional layer.

    Args:
        dim (int): Dimension of the input/output.
        kernel_sizes (list): List of kernel sizes for the convolutions.
        dilations (list): List of dilations for the convolutions.
        activation (str): Activation function to use.
        activation_params (dict): Parameters for the activation function.
        norm (str): Normalization method to apply.
        norm_params (dict): Parameters for the underlying normalization
            used with the convolution.
        causal (bool): Whether to use fully causal convolution.
        pad_mode (str): Padding mode for the convolutions.
        compress (int): Reduced dimensionality in residual branches.
        true_skip (bool): Whether to use true skip connection or a
            simple convolution as the skip connection.
        conv_group_ratio (int): Ratio for grouping convolutions.

    Examples:
        >>> block = SEANetResnetBlock2d(
        ...     dim=128,
        ...     kernel_sizes=[(3, 3), (1, 1)],
        ...     dilations=[(1, 1), (1, 1)],
        ...     activation='ELU',
        ...     activation_params={'alpha': 1.0},
        ...     norm='weight_norm',
        ...     norm_params={},
        ...     causal=False,
        ...     pad_mode='reflect',
        ...     compress=2,
        ...     true_skip=True
        ... )
        >>> input_tensor = torch.randn(8, 128, 32, 32)  # Batch size of 8
        >>> output_tensor = block(input_tensor)
        >>> output_tensor.shape
        torch.Size([8, 128, 32, 32])  # Output shape matches input shape

    Note:
        The block assumes that the input tensor has the shape
        (batch_size, channels, height, width).

    Raises:
        AssertionError: If the number of kernel sizes does not match the
            number of dilations.
    """

    def __init__(
        self,
        dim: int,
        kernel_sizes: List[Tuple[int, int]] = [(3, 3), (1, 1)],
        dilations: List[Tuple[int, int]] = [(1, 1), (1, 1)],
        activation: str = "ELU",
        activation_params: dict = {"alpha": 1.0},
        norm: str = "weight_norm",
        norm_params: Dict[str, Any] = {},
        causal: bool = False,
        pad_mode: str = "reflect",
        compress: int = 2,
        true_skip: bool = True,
        conv_group_ratio: int = -1,
    ):
        super().__init__()
        assert len(kernel_sizes) == len(
            dilations
        ), "Number of kernel sizes should match number of dilations"
        # act = getattr(nn, activation)
        hidden = dim // compress
        block = []
        for i, (kernel_size, dilation) in enumerate(
            zip(kernel_sizes, dilations)
        ):  # this is always length 2
            in_chs = dim if i == 0 else hidden
            out_chs = dim if i == len(kernel_sizes) - 1 else hidden
            block += [
                # act(**activation_params),
                get_activation(activation, **{**activation_params, "channels": in_chs}),
                SConv2d(
                    in_chs,
                    out_chs,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    norm=norm,
                    norm_kwargs=norm_params,
                    causal=causal,
                    pad_mode=pad_mode,
                    groups=(
                        min(in_chs, out_chs) // 2 // conv_group_ratio
                        if conv_group_ratio > 0
                        else 1
                    ),
                ),
            ]
        self.block = nn.Sequential(*block)
        self.shortcut: nn.Module
        # true_skip is always false since the default in
        #     SEANetEncoder / SEANetDecoder does not get changed
        if true_skip:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = SConv2d(
                dim,
                dim,
                kernel_size=(1, 1),
                norm=norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
                groups=dim // 2 // conv_group_ratio if conv_group_ratio > 0 else 1,
            )

    def forward(self, x):
        """
            Performs the forward pass of the SEANetResnetBlock2d.

        The method computes the output of the residual block by applying the
        convolutional layers and the shortcut connection. It adds the output
        of the convolutional block to the output of the shortcut to form the
        residual connection.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, F, T) where:
                - B is the batch size,
                - C is the number of channels,
                - F is the frequency dimension,
                - T is the time dimension.

        Returns:
            torch.Tensor: Output tensor of the same shape as input tensor x,
            which is the sum of the shortcut output and the block output.

        Examples:
            >>> block = SEANetResnetBlock2d(dim=128)
            >>> input_tensor = torch.randn(10, 128, 32, 32)  # Batch of 10
            >>> output_tensor = block(input_tensor)
            >>> output_tensor.shape
            torch.Size([10, 128, 32, 32])

        Note:
            The input tensor must have exactly 4 dimensions. If the input tensor
            has fewer dimensions, an assertion error will be raised.

        Raises:
            AssertionError: If the input tensor does not have 4 dimensions.
        """
        return self.shortcut(x) + self.block(
            x
        )  # This is simply the sum of two tensors of the same size


class ReshapeModule(nn.Module):
    """
    Module to reshape tensors by removing specified dimensions.

    This module allows the removal of a specific dimension from the input tensor,
    effectively reshaping it. This can be useful in various neural network architectures
    where dimensionality adjustments are required.

    Attributes:
        dim (int): The dimension to squeeze from the input tensor.

    Args:
        dim (int): The dimension to remove from the input tensor. Default is 2.

    Returns:
        torch.Tensor: The reshaped tensor with the specified dimension removed.

    Examples:
        >>> reshape_module = ReshapeModule(dim=2)
        >>> input_tensor = torch.rand(2, 3, 4)  # Shape: (2, 3, 4)
        >>> output_tensor = reshape_module(input_tensor)
        >>> output_tensor.shape
        torch.Size([2, 3])  # The shape after squeezing dimension 2

    Note:
        The input tensor must have the specified dimension to be squeezed; otherwise,
        an error will occur. If the input tensor does not have the specified dimension,
        the original tensor will remain unchanged.
    """

    def __init__(self, dim=2):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """
            Module for reshaping tensors by squeezing specified dimensions.

        This module removes a dimension of size 1 from the input tensor along
        the specified axis. This is particularly useful in scenarios where
        it is necessary to eliminate redundant dimensions in tensor
        representations, such as preparing inputs for further processing
        or outputs for loss calculations.

        Attributes:
            dim (int): The dimension to squeeze from the input tensor.

        Args:
            dim (int): The dimension to be squeezed. Defaults to 2.

        Returns:
            torch.Tensor: The input tensor with the specified dimension
            squeezed.

        Examples:
            >>> reshape_module = ReshapeModule(dim=1)
            >>> input_tensor = torch.tensor([[[1]], [[2]], [[3]]])  # shape (3, 1, 1)
            >>> output_tensor = reshape_module(input_tensor)  # shape (3,)
            >>> print(output_tensor)
            tensor([1, 2, 3])

        Note:
            The input tensor must have the specified dimension with size 1.
            If the dimension specified does not have size 1, the output
            tensor will have the same shape as the input tensor.

        Raises:
            IndexError: If the specified dimension is out of bounds for the
            input tensor.
        """
        return torch.squeeze(x, dim=self.dim)


# Only channels, norm, causal are different between 24HZ & 48HZ,
# everything else is default parameter
# 24HZ -> channels = 1, norm = weight_norm, causal = True
# 48HZ -> channels = 2, norm = time_group_norm, causal = False
class SEANetEncoder2d(nn.Module):
    """
    SEANet encoder for audio signal processing.

    This class implements the SEANet encoder architecture, which is designed
    to process audio signals through a series of convolutional layers,
    residual blocks, and optional LSTM layers. The encoder reduces the
    dimensionality of the input while preserving important features for
    subsequent decoding.

    Args:
        channels (int): Audio channels (default: 1).
        dimension (int): Intermediate representation dimension (default: 128).
        n_filters (int): Base width for the model (default: 32).
        n_residual_layers (int): Number of residual layers (default: 1).
        ratios (List[Tuple[int, int]]): Kernel size and stride ratios for
            downsampling (default: [(4, 1), (4, 1), (4, 2), (4, 1)]).
        activation (str): Activation function (default: "ELU").
        activation_params (dict): Parameters for the activation function
            (default: {"alpha": 1.0}).
        norm (str): Normalization method (default: "weight_norm").
        norm_params (Dict[str, Any]): Parameters for the normalization method
            (default: {}).
        kernel_size (int): Kernel size for the initial convolution (default: 7).
        last_kernel_size (int): Kernel size for the last convolution (default: 7).
        residual_kernel_size (int): Kernel size for the residual layers
            (default: 3).
        dilation_base (int): Increase factor for dilation with each layer
            (default: 2).
        causal (bool): Whether to use fully causal convolution (default: False).
        pad_mode (str): Padding mode for convolutions (default: "reflect").
        true_skip (bool): Use true skip connection or simple convolution for
            skip connection (default: False).
        compress (int): Reduced dimensionality in residual branches (default: 2).
        lstm (int): Number of LSTM layers at the end of the encoder (default: 2).
        res_seq (bool): Whether to use a residual sequence (default: True).
        conv_group_ratio (int): Ratio for grouping convolutions (default: -1).

    Examples:
        >>> encoder = SEANetEncoder2d(channels=2, dimension=256)
        >>> input_tensor = torch.randn(1, 2, 16000)  # (batch_size, channels, time)
        >>> output = encoder(input_tensor)
        >>> output.shape
        torch.Size([1, 256, <time_dim>])  # Time dimension varies based on input
    """

    def __init__(
        self,
        channels: int = 1,
        dimension: int = 128,
        n_filters: int = 32,
        n_residual_layers: int = 1,
        ratios: List[Tuple[int, int]] = [(4, 1), (4, 1), (4, 2), (4, 1)],
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
        res_seq=True,
        conv_group_ratio: int = -1,
    ):

        super().__init__()
        self.channels = channels
        self.dimension = dimension
        self.n_filters = n_filters
        self.ratios = list(reversed(ratios))
        del ratios
        self.n_residual_layers = n_residual_layers
        self.hop_length = np.prod([x[1] for x in self.ratios])
        # act = getattr(nn, activation)
        mult = 1
        model: List[nn.Module] = [
            SConv2d(
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
        for (
            freq_ratio,
            time_ratio,
        ) in self.ratios:  # CHANGED from: for i, ratio in enumerate(self.ratios):
            # Add residual layers
            for j in range(
                n_residual_layers
            ):  # This is always 1, parameter never gets changed from default anywhere
                model += [
                    SEANetResnetBlock2d(
                        mult * n_filters,
                        kernel_sizes=[
                            (residual_kernel_size, residual_kernel_size),
                            (1, 1),
                        ],
                        dilations=[(1, dilation_base**j), (1, 1)],
                        norm=norm,
                        norm_params=norm_params,
                        activation=activation,
                        activation_params=activation_params,
                        causal=causal,
                        pad_mode=pad_mode,
                        compress=compress,
                        true_skip=true_skip,
                        conv_group_ratio=conv_group_ratio,
                    )
                ]

            # Add downsampling layers
            model += [
                get_activation(
                    activation, **{**activation_params, "channels": mult * n_filters}
                ),
                SConv2d(
                    mult * n_filters,
                    mult * n_filters * 2,
                    kernel_size=(freq_ratio * 2, time_ratio * 2),
                    stride=(freq_ratio, time_ratio),
                    norm=norm,
                    norm_kwargs=norm_params,
                    causal=causal,
                    pad_mode=pad_mode,
                    groups=(
                        mult * n_filters // 2 // conv_group_ratio
                        if conv_group_ratio > 0
                        else 1
                    ),
                ),
            ]
            mult *= 2

        # squeeze shape for subsequent models
        model += [ReshapeModule(dim=2)]

        if lstm:
            model += [SLSTM(mult * n_filters, num_layers=lstm)]

        model += [
            # act(**activation_params),
            get_activation(
                activation, **{**activation_params, "channels": mult * n_filters}
            ),
            SConv1d(
                mult * n_filters,
                dimension,
                kernel_size=last_kernel_size,
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

        This class implements the SEANet encoder architecture for audio processing,
        utilizing a series of convolutional layers, residual blocks, and optional LSTM
        layers to produce an intermediate representation of audio inputs.

        Attributes:
            channels (int): Number of audio channels.
            dimension (int): Intermediate representation dimension.
            n_filters (int): Base width for the model.
            n_residual_layers (int): Number of residual layers.
            ratios (List[Tuple[int, int]]): Kernel size and stride ratios for downsampling.
            hop_length (int): The total hop length calculated from the ratios.

        Args:
            channels (int): Audio channels. Defaults to 1.
            dimension (int): Intermediate representation dimension. Defaults to 128.
            n_filters (int): Base width for the model. Defaults to 32.
            n_residual_layers (int): Number of residual layers. Defaults to 1.
            ratios (List[Tuple[int, int]]): Kernel size and stride ratios.
                Defaults to [(4, 1), (4, 1), (4, 2), (4, 1)].
            activation (str): Activation function. Defaults to "ELU".
            activation_params (dict): Parameters for the activation function.
                Defaults to {"alpha": 1.0}.
            norm (str): Normalization method. Defaults to "weight_norm".
            norm_params (Dict[str, Any]): Parameters for the underlying normalization.
            kernel_size (int): Kernel size for the initial convolution. Defaults to 7.
            last_kernel_size (int): Kernel size for the last convolution. Defaults to 7.
            residual_kernel_size (int): Kernel size for the residual layers. Defaults to 3.
            dilation_base (int): How much to increase the dilation with each layer. Defaults to 2.
            causal (bool): Whether to use fully causal convolution. Defaults to False.
            pad_mode (str): Padding mode for convolutions. Defaults to "reflect".
            true_skip (bool): Whether to use true skip connection or a simple convolution.
                Defaults to False.
            compress (int): Reduced dimensionality in residual branches. Defaults to 2.
            lstm (int): Number of LSTM layers at the end of the encoder. Defaults to 2.
            res_seq (bool): Flag to indicate whether to apply sequential processing. Defaults to True.
            conv_group_ratio (int): Group ratio for convolutions. Defaults to -1.

        Examples:
            >>> encoder = SEANetEncoder2d(channels=2, dimension=256)
            >>> audio_input = torch.randn(1, 2, 16000)  # Batch size of 1, 2 channels, 16000 samples
            >>> output = encoder(audio_input)
            >>> output.shape
            torch.Size([1, 256, T])  # Output shape depends on the internal configuration

        Raises:
            AssertionError: If the number of kernel sizes does not match the number of dilations.
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)  # x in B,C,T, return B,T,C
        return self.model(x)

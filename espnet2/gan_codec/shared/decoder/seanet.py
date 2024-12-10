# Adapted from https://github.com/facebookresearch/encodec by Jiatong Shi

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in https://github.com/facebookresearch/encodec/tree/main

"""Encodec SEANet-based encoder and decoder implementation."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm, weight_norm

from espnet2.gan_codec.shared.encoder.seanet import (
    SLSTM,
    SConv1d,
    SEANetResnetBlock,
    apply_parametrization_norm,
    get_norm_module,
)
from espnet2.gan_codec.shared.encoder.snake_activation import Snake1d


def unpad1d(x: torch.Tensor, paddings: Tuple[int, int]):
    """
    Remove padding from a 1D tensor, handling zero padding properly.

    This function is designed to remove the specified padding from a 1D
    tensor. It is important to ensure that the padding values are non-negative
    and that their sum does not exceed the length of the tensor.

    Args:
        x (torch.Tensor): The input tensor from which padding will be removed.
        paddings (Tuple[int, int]): A tuple specifying the left and right
            padding to remove from the tensor.

    Returns:
        torch.Tensor: The tensor with the specified padding removed.

    Raises:
        AssertionError: If padding values are negative or their sum exceeds
            the length of the tensor.

    Examples:
        >>> import torch
        >>> x = torch.tensor([1, 2, 3, 4, 5])
        >>> unpad1d(x, (1, 1))
        tensor([2, 3, 4])
        >>> unpad1d(x, (0, 2))
        tensor([1, 2, 3])
    """
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    assert (padding_left + padding_right) <= x.shape[-1]
    end = x.shape[-1] - padding_right
    return x[..., padding_left:end]


class NormConvTranspose1d(nn.Module):
    """
    Wrapper around ConvTranspose1d with normalization for uniformity.

    This class encapsulates the functionality of a 1D transposed convolution
    (also known as a deconvolution) and applies a normalization technique to
    ensure consistency across various normalization approaches.

    Attributes:
        convtr (nn.Module): The transposed convolution layer.
        norm (nn.Module): The normalization layer applied to the transposed
            convolution output.
        norm_type (str): The type of normalization used.

    Args:
        *args: Variable length argument list for the transposed convolution.
        causal (bool): If True, the convolution is causal (i.e., it only
            considers past inputs). Defaults to False.
        norm (str): The normalization method to apply. Defaults to "none".
        norm_kwargs (Dict[str, Any]): Additional keyword arguments for the
            normalization layer.
        **kwargs: Additional keyword arguments for the transposed convolution.

    Examples:
        >>> layer = NormConvTranspose1d(in_channels=16, out_channels=33,
        ...                              kernel_size=3, stride=2, norm='batch_norm')
        >>> input_tensor = torch.randn(1, 16, 50)
        >>> output_tensor = layer(input_tensor)
        >>> output_tensor.shape
        torch.Size([1, 33, 98])

    Raises:
        ValueError: If any of the arguments passed to the convolution layer
            are invalid.
    """

    def __init__(
        self,
        *args,
        causal: bool = False,
        norm: str = "none",
        norm_kwargs: Dict[str, Any] = {},
        **kwargs
    ):
        super().__init__()
        self.convtr = apply_parametrization_norm(
            nn.ConvTranspose1d(*args, **kwargs), norm
        )
        self.norm = get_norm_module(self.convtr, causal, norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x):
        """
            Applies the transposed convolution followed by normalization to the input.

        Args:
            x (torch.Tensor): The input tensor to the transposed convolution. It is
                expected to have shape (batch_size, in_channels, length).

        Returns:
            torch.Tensor: The output tensor after applying the transposed convolution
            and normalization. The shape of the output will depend on the parameters
            of the convolution.

        Examples:
            >>> model = NormConvTranspose1d(in_channels=16, out_channels=32,
            ...                              kernel_size=3, stride=2)
            >>> input_tensor = torch.randn(8, 16, 10)  # batch_size=8, in_channels=16
            >>> output_tensor = model(input_tensor)
            >>> output_tensor.shape
            torch.Size([8, 32, 20])  # Output shape after transposed convolution

        Note:
            The normalization method applied can be specified during the initialization
            of the class. This method can handle various normalization techniques
            seamlessly, providing a uniform interface for different approaches.

        Raises:
            ValueError: If the input tensor does not have the expected shape.
        """
        x = self.convtr(x)
        x = self.norm(x)
        return x


class SConvTranspose1d(nn.Module):
    """
        SConvTranspose1d is a 1D transposed convolution layer that incorporates built-in
    handling of asymmetric or causal padding and normalization. This class is
    designed to provide a consistent interface for various normalization approaches
    while managing the complexities of transposed convolutions.

    Attributes:
        convtr (NormConvTranspose1d): The wrapped transposed convolution layer with
            normalization applied.
        causal (bool): Indicates if the convolution is causal.
        trim_right_ratio (float): Ratio for trimming the output on the right side.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        causal (bool, optional): If True, use causal convolution. Defaults to False.
        norm (str, optional): Normalization method. Defaults to "none".
        trim_right_ratio (float, optional): Ratio for trimming at the right of the
            transposed convolution. Defaults to 1.0.
        norm_kwargs (Dict[str, Any], optional): Additional parameters for the
            normalization module. Defaults to an empty dictionary.

    Raises:
        AssertionError: If `trim_right_ratio` is not in the range [0.0, 1.0] or if
        `trim_right_ratio` is not equal to 1.0 when causal is False.

    Examples:
        >>> layer = SConvTranspose1d(in_channels=16, out_channels=32,
        ...                           kernel_size=3, stride=2, causal=True)
        >>> input_tensor = torch.randn(10, 16, 50)  # Batch size 10, 16 channels, length 50
        >>> output_tensor = layer(input_tensor)
        >>> print(output_tensor.shape)  # Output shape will depend on kernel size and stride

    Note:
        The `trim_right_ratio` should be set to a value between 0.0 and 1.0. When
        `causal` is set to True, the trimming will be applied to the right side
        of the output tensor according to the specified ratio.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        causal: bool = False,
        norm: str = "none",
        trim_right_ratio: float = 1.0,
        norm_kwargs: Dict[str, Any] = {},
    ):
        super().__init__()
        self.convtr = NormConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            causal=causal,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )
        self.causal = causal
        self.trim_right_ratio = trim_right_ratio
        assert (
            self.causal or self.trim_right_ratio == 1.0
        ), "`trim_right_ratio` != 1.0 only makes sense for causal convolutions"
        assert self.trim_right_ratio >= 0.0 and self.trim_right_ratio <= 1.0

    def forward(self, x):
        """
            Forward pass for the SEANetDecoder.

        This method takes an input tensor `z` and passes it through the decoder
        network, applying a series of convolutions and activations to produce
        the output.

        Args:
            z (torch.Tensor): The input tensor, typically representing the
                encoded audio signal with shape (batch_size, channels, length).

        Returns:
            torch.Tensor: The output tensor after passing through the decoder,
                representing the reconstructed audio signal.

        Examples:
            >>> decoder = SEANetDecoder(channels=1)
            >>> input_tensor = torch.randn(16, 32, 100)  # Example input
            >>> output_tensor = decoder(input_tensor)
            >>> print(output_tensor.shape)
            torch.Size([16, 1, <length>])  # Output shape will depend on the
                                             # specific configuration of the model.

        Note:
            The output shape will depend on the architecture defined during
            the initialization of the SEANetDecoder, particularly the
            kernel sizes, strides, and padding applied throughout the model.

        Raises:
            ValueError: If the input tensor `z` does not have the expected
                shape or type.
        """
        kernel_size = self.convtr.convtr.kernel_size[0]
        stride = self.convtr.convtr.stride[0]
        padding_total = kernel_size - stride

        y = self.convtr(x)

        # We will only trim fixed padding. Extra padding from
        # `pad_for_conv1d` would be removed at the very end,
        # when keeping only the right length for the output, as
        # removing it here would require also passing the length
        # at the matching layer in the encoder.
        if self.causal:
            # Trim the padding on the right according to the specified ratio
            # if trim_right_ratio = 1.0, trim everything from right
            padding_right = math.ceil(padding_total * self.trim_right_ratio)
            padding_left = padding_total - padding_right
            y = unpad1d(y, (padding_left, padding_right))
        else:
            # Asymmetric padding required for odd strides
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            y = unpad1d(y, (padding_left, padding_right))
        return y


class SEANetDecoder(nn.Module):
    """
    SEANet decoder.

    This class implements a SEANet-based decoder for audio processing using
    transposed convolutions and optional LSTM layers. It allows for
    customizable activation functions, normalization methods, and
    residual connections.

    Args:
        channels (int): Audio channels.
        dimension (int): Intermediate representation dimension.
        n_filters (int): Base width for the model.
        n_residual_layers (int): Number of residual layers.
        ratios (Sequence[int]): Kernel size and stride ratios.
        activation (str): Activation function.
        activation_params (dict): Parameters to provide to the activation function.
        final_activation (Optional[str]): Final activation function after all
            convolutions.
        final_activation_params (Optional[dict]): Parameters to provide to the
            final activation function.
        norm (str): Normalization method.
        norm_params (dict): Parameters to provide to the underlying normalization
            used along with the convolution.
        kernel_size (int): Kernel size for the initial convolution.
        last_kernel_size (int): Kernel size for the final convolution.
        residual_kernel_size (int): Kernel size for the residual layers.
        dilation_base (int): How much to increase the dilation with each layer.
        causal (bool): Whether to use fully causal convolution.
        pad_mode (str): Padding mode for the convolutions.
        true_skip (bool): Whether to use true skip connection or a simple
            (streamable) convolution as the skip connection in the residual
            network blocks.
        compress (int): Reduced dimensionality in residual branches (from
            Demucs v3).
        lstm (int): Number of LSTM layers at the end of the encoder.
        trim_right_ratio (float): Ratio for trimming at the right of the
            transposed convolution under the causal setup. If equal to 1.0,
            it means that all the trimming is done at the right.

    Examples:
        >>> decoder = SEANetDecoder(channels=1, dimension=128, n_filters=32)
        >>> input_tensor = torch.randn(1, 32, 256)  # (batch_size, channels, length)
        >>> output_tensor = decoder(input_tensor)
        >>> print(output_tensor.shape)  # Output shape will depend on configuration
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
        final_activation: Optional[str] = None,
        final_activation_params: Optional[dict] = None,
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
        trim_right_ratio: float = 1.0,
    ):
        super().__init__()
        self.dimension = dimension
        self.channels = channels
        self.n_filters = n_filters
        self.ratios = ratios
        del ratios
        self.n_residual_layers = n_residual_layers
        self.hop_length = np.prod(self.ratios)

        if activation == "Snake":
            act = Snake1d
        else:
            act = getattr(nn, activation)
        mult = int(2 ** len(self.ratios))
        model: List[nn.Module] = [
            SConv1d(
                dimension,
                mult * n_filters,
                kernel_size,
                norm=norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            )
        ]

        if lstm:
            model += [SLSTM(mult * n_filters, num_layers=lstm)]

        # Upsample to raw audio scale
        for i, ratio in enumerate(self.ratios):
            # Add upsampling layers
            model += [
                act(**activation_params),
                SConvTranspose1d(
                    mult * n_filters,
                    mult * n_filters // 2,
                    kernel_size=ratio * 2,
                    stride=ratio,
                    norm=norm,
                    norm_kwargs=norm_params,
                    causal=causal,
                    trim_right_ratio=trim_right_ratio,
                ),
            ]
            # Add residual layers
            for j in range(n_residual_layers):
                model += [
                    SEANetResnetBlock(
                        mult * n_filters // 2,
                        kernel_sizes=[residual_kernel_size, 1],
                        dilations=[dilation_base**j, 1],
                        activation=activation,
                        activation_params=activation_params,
                        norm=norm,
                        norm_params=norm_params,
                        causal=causal,
                        pad_mode=pad_mode,
                        compress=compress,
                        true_skip=true_skip,
                    )
                ]

            mult //= 2

        # Add final layers
        model += [
            act(**activation_params),
            SConv1d(
                n_filters,
                channels,
                last_kernel_size,
                norm=norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            ),
        ]
        # Add optional final activation to decoder (eg. tanh)
        if final_activation is not None:
            final_act = getattr(nn, final_activation)
            final_activation_params = final_activation_params or {}
            model += [final_act(**final_activation_params)]
        self.model = nn.Sequential(*model)

    def forward(self, z):
        """
            SEANet decoder.

        This class implements the SEANet decoder, which is designed for audio
        processing tasks. It employs a series of convolutional and residual layers
        to decode an intermediate representation into an audio signal.

        Args:
            channels (int): Audio channels.
            dimension (int): Intermediate representation dimension.
            n_filters (int): Base width for the model.
            n_residual_layers (int): Number of residual layers.
            ratios (Sequence[int]): Kernel size and stride ratios.
            activation (str): Activation function.
            activation_params (dict): Parameters to provide to the activation function.
            final_activation (str): Final activation function after all convolutions.
            final_activation_params (dict): Parameters to provide to the activation
                function.
            norm (str): Normalization method.
            norm_params (dict): Parameters to provide to the underlying normalization
                used along with the convolution.
            kernel_size (int): Kernel size for the initial convolution.
            last_kernel_size (int): Kernel size for the last convolution.
            residual_kernel_size (int): Kernel size for the residual layers.
            dilation_base (int): How much to increase the dilation with each layer.
            causal (bool): Whether to use fully causal convolution.
            pad_mode (str): Padding mode for the convolutions.
            true_skip (bool): Whether to use true skip connection or a simple
                (streamable) convolution as the skip connection in the residual
                network blocks.
            compress (int): Reduced dimensionality in residual branches (from
                Demucs v3).
            lstm (int): Number of LSTM layers at the end of the encoder.
            trim_right_ratio (float): Ratio for trimming at the right of the
                transposed convolution under the causal setup. If equal to 1.0, it
                means that all the trimming is done at the right.

        Examples:
            >>> decoder = SEANetDecoder(channels=1, dimension=128)
            >>> input_tensor = torch.randn(1, 128, 100)  # Example input
            >>> output_tensor = decoder(input_tensor)
            >>> output_tensor.shape
            torch.Size([1, 1, <output_length>])  # Output length depends on config

        Returns:
            torch.Tensor: The decoded audio signal as a tensor.
        """
        y = self.model(z)
        return y

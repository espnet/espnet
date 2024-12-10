# Adapted by Yihan Wu for 2D SEANet (from seanet.py)

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in https://github.com/facebookresearch/encodec/tree/main

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from espnet2.gan_codec.shared.encoder.seanet import (
    SLSTM,
    SConv1d,
    apply_parametrization_norm,
    get_extra_padding_for_conv1d,
    get_norm_module,
)
from espnet2.gan_codec.shared.encoder.seanet_2d import SConv2d, get_activation


def unpad2d(x: torch.Tensor, paddings: Tuple[Tuple[int, int], Tuple[int, int]]):
    """
    Remove padding from a 2D tensor, ensuring proper handling of zero padding.

    This function takes a 2D tensor `x` and a tuple of padding values, and returns
    the tensor with the specified paddings removed. It ensures that the padding values
    are valid and that the resulting dimensions are appropriate after unpadding.

    Args:
        x (torch.Tensor): The input tensor of shape (..., height, width).
        paddings (Tuple[Tuple[int, int], Tuple[int, int]]): A tuple specifying the
            paddings to be removed. The first element corresponds to the
            (top, bottom) padding for the height dimension, and the second
            element corresponds to the (left, right) padding for the width
            dimension.

    Returns:
        torch.Tensor: The tensor with the specified padding removed.

    Raises:
        AssertionError: If any padding values are negative or exceed the dimensions
        of the tensor.

    Examples:
        >>> import torch
        >>> x = torch.randn(1, 3, 10, 10)  # A random tensor of shape (1, 3, 10, 10)
        >>> paddings = ((2, 2), (3, 3))  # Remove 2 rows from top and bottom,
        ...                               # and 3 columns from left and right
        >>> unpadded_x = unpad2d(x, paddings)
        >>> unpadded_x.shape
        torch.Size([1, 3, 6, 4])  # The resulting shape after unpadding
    """
    padding_time_left, padding_time_right = paddings[0]
    padding_freq_left, padding_freq_right = paddings[1]
    assert min(paddings[0]) >= 0 and min(paddings[1]) >= 0, paddings
    assert (padding_time_left + padding_time_right) <= x.shape[-1] and (
        padding_freq_left + padding_freq_right
    ) <= x.shape[-2]

    freq_end = x.shape[-2] - padding_freq_right
    time_end = x.shape[-1] - padding_time_right
    return x[..., padding_freq_left:freq_end, padding_time_left:time_end]


class NormConvTranspose2d(nn.Module):
    """
    Wrapper around ConvTranspose2d with normalization applied to provide a
    uniform interface across normalization approaches.

    This class allows for the easy integration of various normalization
    techniques into a transposed convolution layer, which can be useful in
    deep learning architectures where normalization can stabilize training
    and improve convergence.

    Attributes:
        convtr (nn.Module): The ConvTranspose2d layer with applied
            normalization.
        norm (nn.Module): The normalization layer applied to the output of
            the transposed convolution.

    Args:
        *args: Variable length argument list for ConvTranspose2d.
        causal (bool): Whether to apply causal convolution. Defaults to False.
        norm (str): Type of normalization to apply. Options include "none",
            "batch_norm", "layer_norm", etc. Defaults to "none".
        norm_kwargs (Dict[str, Any]): Additional keyword arguments for the
            normalization layer.

    Returns:
        Tensor: Output tensor after applying the transposed convolution and
        normalization.

    Examples:
        >>> layer = NormConvTranspose2d(in_channels=16, out_channels=33,
        ...                              kernel_size=(3, 3), stride=(2, 2),
        ...                              norm='batch_norm')
        >>> input_tensor = torch.randn(1, 16, 8, 8)
        >>> output_tensor = layer(input_tensor)
        >>> output_tensor.shape
        torch.Size([1, 33, 16, 16])
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
            nn.ConvTranspose2d(*args, **kwargs), norm
        )
        self.norm = get_norm_module(self.convtr, causal, norm, **norm_kwargs)

    def forward(self, x):
        """
            Applies the transposed convolution and normalization to the input tensor.

        This method takes an input tensor `x`, applies a transposed convolution
        followed by a normalization operation. The transposed convolution is defined
        by the parameters passed during the initialization of the `NormConvTranspose2d`
        class.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W), where:
                - N is the batch size,
                - C is the number of input channels,
                - H is the height of the input,
                - W is the width of the input.

        Returns:
            torch.Tensor: The output tensor after applying the transposed convolution
            and normalization, with the same shape as the input tensor.

        Examples:
            >>> norm_conv_transpose = NormConvTranspose2d(in_channels=3,
            ...                                             out_channels=6,
            ...                                             kernel_size=(3, 3))
            >>> input_tensor = torch.randn(1, 3, 8, 8)  # Example input
            >>> output_tensor = norm_conv_transpose(input_tensor)
            >>> output_tensor.shape
            torch.Size([1, 6, 10, 10])  # Shape will depend on kernel_size and stride

        Note:
            Ensure that the input tensor `x` has the correct number of dimensions
            (4D) as expected by the transposed convolution layer.
        """
        x = self.convtr(x)
        x = self.norm(x)
        return x


class SConvTranspose2d(nn.Module):
    """
    ConvTranspose2d with built-in handling of asymmetric or causal padding
    and normalization.

    Note:
        Causal padding only makes sense on the time (the last) axis.
        The frequency (the second last) axis is always non-causally padded.

    Attributes:
        convtr (NormConvTranspose2d): The convolutional transpose layer with
            normalization.
        out_padding (List[Tuple[int, int]]): Padding to be added to the output.
        causal (bool): Whether to use causal padding.
        trim_right_ratio (float): Ratio for trimming the output on the right side.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (Union[int, Tuple[int, int]]): Size of the convolving kernel.
        stride (Union[int, Tuple[int, int]], optional): Stride of the convolution.
            Defaults to 1.
        causal (bool, optional): If True, use causal padding. Defaults to False.
        norm (str, optional): Type of normalization to apply. Defaults to "none".
        trim_right_ratio (float, optional): Ratio for trimming at the right of
            the transposed convolution under the causal setup. Defaults to 1.0.
        norm_kwargs (Dict[str, Any], optional): Additional arguments for normalization.
        out_padding (Union[int, List[Tuple[int, int]]], optional): Padding added to
            the output. Defaults to 0.
        groups (int, optional): Number of blocked connections from input channels
            to output channels. Defaults to 1.

    Raises:
        AssertionError: If `trim_right_ratio` is not 1.0 and `causal` is False.
        AssertionError: If `trim_right_ratio` is not between 0.0 and 1.0.

    Examples:
        >>> layer = SConvTranspose2d(in_channels=16, out_channels=33,
        ...                           kernel_size=(3, 3), stride=(2, 2),
        ...                           causal=True, trim_right_ratio=0.5)
        >>> input_tensor = torch.randn(1, 16, 10, 10)
        >>> output_tensor = layer(input_tensor)
        >>> output_tensor.shape
        torch.Size([1, 33, 20, 20])
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        causal: bool = False,
        norm: str = "none",
        trim_right_ratio: float = 1.0,
        norm_kwargs: Dict[str, Any] = {},
        out_padding: Union[int, List[Tuple[int, int]]] = 0,
        groups: int = 1,
    ):
        super().__init__()
        self.convtr = NormConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            causal=causal,
            norm=norm,
            norm_kwargs=norm_kwargs,
            groups=groups,
        )
        if isinstance(out_padding, int):
            self.out_padding = [(out_padding, out_padding), (out_padding, out_padding)]
        else:
            self.out_padding = out_padding
        self.causal = causal
        self.trim_right_ratio = trim_right_ratio
        assert (
            self.causal or self.trim_right_ratio == 1.0
        ), "`trim_right_ratio` != 1.0 only makes sense for causal convolutions"
        assert self.trim_right_ratio >= 0.0 and self.trim_right_ratio <= 1.0

    def forward(self, x):
        """
            Applies the transposed convolution and normalization to the input tensor.

        This method first applies a transposed convolution operation followed by a
        normalization step. The transposed convolution is performed using the
        `ConvTranspose2d` layer, and the normalization is applied based on the
        specified normalization method during the initialization of the class.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W), where N is the
                batch size, C is the number of input channels, H is the height,
                and W is the width of the input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the transposed convolution
            and normalization. The output tensor will have the same shape as the
            input tensor (N, C, H, W) if the parameters are set accordingly.

        Examples:
            >>> model = SConvTranspose2d(in_channels=1, out_channels=1, kernel_size=3)
            >>> input_tensor = torch.randn(1, 1, 64, 64)  # Example input
            >>> output_tensor = model(input_tensor)
            >>> print(output_tensor.shape)  # Output shape will be (1, 1, 64, 64)

        Note:
            - Ensure that the input tensor `x` has the correct shape as specified.
            - The normalization method and its parameters can be adjusted during
            initialization of the class.
        """
        kernel_size = self.convtr.convtr.kernel_size[0]
        stride = self.convtr.convtr.stride[0]
        padding_freq_total = kernel_size - stride
        kernel_size = self.convtr.convtr.kernel_size[1]
        stride = self.convtr.convtr.stride[1]
        padding_time_total = kernel_size - stride

        y = self.convtr(x)

        # We will only trim fixed padding. Extra padding from `pad_for_conv1d`
        # would be removed at the very end, when keeping only the right length
        # for the output, as removing it here would require also passing the
        # length at the matching layer in the encoder.
        (freq_out_pad_left, freq_out_pad_right) = self.out_padding[0]
        (time_out_pad_left, time_out_pad_right) = self.out_padding[1]
        if self.causal:
            # Trim the padding on the right according to the specified ratio
            # if trim_right_ratio = 1.0, trim everything from right
            padding_freq_right = padding_freq_total // 2
            padding_freq_left = padding_freq_total - padding_freq_right
            padding_time_right = math.ceil(padding_time_total * self.trim_right_ratio)
            padding_time_left = padding_time_total - padding_time_right
            y = unpad2d(
                y,
                (
                    (
                        max(padding_time_left - time_out_pad_left, 0),
                        max(padding_time_right - time_out_pad_right, 0),
                    ),
                    (
                        max(padding_freq_left - freq_out_pad_left, 0),
                        max(padding_freq_right - freq_out_pad_right, 0),
                    ),
                ),
            )
        else:
            # Asymmetric padding required for odd strides
            padding_freq_right = padding_freq_total // 2
            padding_freq_left = padding_freq_total - padding_freq_right
            padding_time_right = padding_time_total // 2
            padding_time_left = padding_time_total - padding_time_right
            y = unpad2d(
                y,
                (
                    (
                        max(padding_time_left - time_out_pad_left, 0),
                        max(padding_time_right - time_out_pad_right, 0),
                    ),
                    (
                        max(padding_freq_left - freq_out_pad_left, 0),
                        max(padding_freq_right - freq_out_pad_right, 0),
                    ),
                ),
            )
        return y


class SEANetResnetBlock2d(nn.Module):
    """
    Residual block from the SEANet model.

    This class implements a residual block that includes convolutional layers
    and optional normalization and activation functions. The architecture allows
    for flexible configuration through various parameters, including kernel sizes,
    dilations, and normalization methods.

    Attributes:
        block (nn.Sequential): Sequential container of activation and convolution
            layers.
        shortcut (nn.Module): A skip connection that can either be an identity
            mapping or a convolution layer, depending on the `true_skip` parameter.

    Args:
        dim (int): Dimension of the input/output.
        kernel_sizes (list): List of kernel sizes for the convolutions.
        dilations (list): List of dilations for the convolutions.
        activation (str): Activation function to use.
        activation_params (dict): Parameters for the activation function.
        norm (str): Normalization method to apply.
        norm_params (dict): Parameters for the normalization used along with the
            convolution.
        causal (bool): Whether to use fully causal convolution.
        pad_mode (str): Padding mode for the convolutions.
        compress (int): Reduced dimensionality in residual branches (from Demucs v3).
        true_skip (bool): Whether to use true skip connection or a simple
            convolution as the skip connection.
        conv_group_ratio (int): Ratio for grouping channels in convolutions.

    Examples:
        >>> block = SEANetResnetBlock2d(dim=64)
        >>> input_tensor = torch.randn(1, 64, 128, 128)  # (batch, channels, height, width)
        >>> output_tensor = block(input_tensor)
        >>> output_tensor.shape
        torch.Size([1, 64, 128, 128])  # Output has the same shape as input

    Raises:
        AssertionError: If the number of kernel sizes does not match the number
        of dilations.
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
        # SEANetEncoder / SEANetDecoder does not get changed
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
            Forward pass for the SEANetResnetBlock2d module.

        This method computes the output of the residual block by applying a
        series of convolutional layers followed by a shortcut connection. The
        input tensor `x` is passed through the convolutional block and the
        output is added to the shortcut output.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W) where B is
                the batch size, C is the number of channels, H is the height,
                and W is the width of the input tensor.

        Returns:
            torch.Tensor: Output tensor of the same shape as the input tensor.

        Examples:
            >>> block = SEANetResnetBlock2d(dim=64)
            >>> input_tensor = torch.randn(1, 64, 32, 32)  # Example input
            >>> output_tensor = block(input_tensor)
            >>> print(output_tensor.shape)  # Should be torch.Size([1, 64, 32, 32])

        Note:
            The shortcut connection is implemented as an identity operation
            by default. If `true_skip` is set to False, a 1x1 convolution is
            used instead.
        """
        return self.shortcut(x) + self.block(
            x
        )  # This is simply the sum of two tensors of the same size


class ReshapeModule(nn.Module):
    """
    Module for reshaping tensors by adding an extra dimension.

    This module is designed to add a specified dimension to the input tensor
    using `torch.unsqueeze()`. It can be useful in scenarios where a tensor
    needs to be reshaped for further processing in a neural network.

    Attributes:
        dim (int): The dimension index at which to add the new dimension.
                   Default is 2.

    Args:
        dim (int): The dimension index to add. Defaults to 2.

    Returns:
        torch.Tensor: The input tensor with an additional dimension.

    Examples:
        >>> reshape_module = ReshapeModule(dim=1)
        >>> input_tensor = torch.tensor([[1, 2], [3, 4]])
        >>> output_tensor = reshape_module(input_tensor)
        >>> output_tensor.shape
        torch.Size([2, 1, 2])  # A new dimension is added at index 1.

    Note:
        The input tensor should have a shape compatible with the added
        dimension index. If the dimension index is out of range, it may
        lead to unexpected results or errors.
    """

    def __init__(self, dim=2):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """
            Module to reshape the input tensor by adding a new dimension.

        This module adds a new dimension to the input tensor at the specified
        dimension index. It is useful for manipulating tensor shapes for
        subsequent processing in neural network architectures.

        Attributes:
            dim (int): The dimension index at which to insert the new dimension.

        Args:
            dim (int): The dimension index for the new dimension. Defaults to 2.

        Returns:
            torch.Tensor: The input tensor with an additional dimension added
            at the specified index.

        Examples:
            >>> reshape_module = ReshapeModule(dim=1)
            >>> input_tensor = torch.randn(2, 3, 4)  # Shape: (2, 3, 4)
            >>> output_tensor = reshape_module(input_tensor)
            >>> output_tensor.shape
            torch.Size([2, 1, 3, 4])  # New shape with added dimension at index 1

        Note:
            This module is designed for use in neural network pipelines where
            reshaping of tensors is required.
        """
        return torch.unsqueeze(x, dim=self.dim)


class SEANetDecoder2d(nn.Module):
    """
    SEANet decoder for audio signal processing.

    This class implements the SEANet decoder architecture, which is designed to
    decode intermediate representations into audio signals. The decoder consists
    of a series of convolutional and residual layers, with optional normalization
    and activation functions applied throughout the network.

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
        compress (int): Reduced dimensionality in residual branches (from Demucs v3).
        lstm (int): Number of LSTM layers at the end of the encoder.
        trim_right_ratio (float): Ratio for trimming at the right of the transposed
            convolution under the causal setup. If equal to 1.0, it means that
            all the trimming is done at the right.

    Examples:
        >>> decoder = SEANetDecoder2d(channels=1, dimension=128, n_filters=32)
        >>> input_tensor = torch.randn(10, 128, 64)  # Batch of 10, 128 channels, 64 time steps
        >>> output = decoder(input_tensor)
        >>> print(output.shape)
        torch.Size([10, 1, T])  # Output shape will depend on the model configuration

    Attributes:
        model (nn.Sequential): The sequential model composed of layers defined
            in the constructor.

    Note:
        The final activation is optional and defaults to None.
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
        res_seq=True,
        last_out_padding: List[Union[int, int]] = [(0, 1), (0, 0)],
        tr_conv_group_ratio: int = -1,
        conv_group_ratio: int = -1,
    ):
        super().__init__()
        self.dimension = dimension
        self.channels = channels
        self.n_filters = n_filters
        self.ratios = ratios
        del ratios
        self.n_residual_layers = n_residual_layers
        self.hop_length = np.prod([x[1] for x in self.ratios])

        # act = getattr(nn, activation)
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

        model += [SLSTM(mult * n_filters, num_layers=lstm)]

        model += [ReshapeModule(dim=2)]

        # Upsample to raw audio scale
        for i, (freq_ratio, time_ratio) in enumerate(self.ratios):
            # Add upsampling layers
            model += [
                # act(**activation_params),
                get_activation(
                    activation, **{**activation_params, "channels": mult * n_filters}
                ),
                SConvTranspose2d(
                    mult * n_filters,
                    mult * n_filters // 2,
                    kernel_size=(freq_ratio * 2, time_ratio * 2),
                    stride=(freq_ratio, time_ratio),
                    norm=norm,
                    norm_kwargs=norm_params,
                    causal=causal,
                    trim_right_ratio=trim_right_ratio,
                    out_padding=last_out_padding if i == len(self.ratios) - 1 else 0,
                    groups=(
                        mult * n_filters // 2 // tr_conv_group_ratio
                        if tr_conv_group_ratio > 0
                        else 1
                    ),
                ),
            ]
            # Add residual layers
            for j in range(n_residual_layers):
                model += [
                    SEANetResnetBlock2d(
                        mult * n_filters // 2,
                        kernel_sizes=[
                            (residual_kernel_size, residual_kernel_size),
                            (1, 1),
                        ],
                        dilations=[(1, dilation_base**j), (1, 1)],
                        activation=activation,
                        activation_params=activation_params,
                        norm=norm,
                        norm_params=norm_params,
                        causal=causal,
                        pad_mode=pad_mode,
                        compress=compress,
                        true_skip=true_skip,
                        conv_group_ratio=conv_group_ratio,
                    )
                ]
            mult //= 2

        # Add final layers
        model += [
            # act(**activation_params),
            get_activation(activation, **{**activation_params, "channels": n_filters}),
            SConv2d(
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
        if final_activation is not None:  # This is always None
            final_act = getattr(nn, final_activation)
            final_activation_params = final_activation_params or {}
            model += [final_act(**final_activation_params)]
        self.model = nn.Sequential(*model)

    def output_size(self):
        """
            Returns the number of output channels for the SEANet decoder.

        This method provides the output size of the decoder, which corresponds
        to the number of audio channels that the decoder will produce. It is
        primarily used to configure the final layer of the decoder to ensure
        that the output shape matches the expected audio channel format.

        Returns:
            int: The number of output channels of the decoder.

        Examples:
            decoder = SEANetDecoder2d(channels=2)
            output_channels = decoder.output_size()
            print(output_channels)  # Output: 2
        """
        return self.channels

    def forward(self, z):
        """
            SEANet decoder for audio signal reconstruction.

        This class implements a decoder based on the SEANet architecture,
        designed to convert latent representations back into audio signals.

        Args:
            channels (int): Audio channels.
            dimension (int): Intermediate representation dimension.
            n_filters (int): Base width for the model.
            n_residual_layers (int): Number of residual layers.
            ratios (Sequence[int]): Kernel size and stride ratios.
            activation (str): Activation function.
            activation_params (dict): Parameters to provide to the activation function.
            final_activation (str): Final activation function after all convolutions.
            final_activation_params (dict): Parameters to provide to the activation function.
            norm (str): Normalization method.
            norm_params (dict): Parameters to provide to the underlying normalization
                used along with the convolution.
            kernel_size (int): Kernel size for the initial convolution.
            last_kernel_size (int): Kernel size for the final convolution.
            residual_kernel_size (int): Kernel size for the residual layers.
            dilation_base (int): How much to increase the dilation with each layer.
            causal (bool): Whether to use fully causal convolution.
            pad_mode (str): Padding mode for the convolutions.
            true_skip (bool): Whether to use true skip connection or a simple (streamable)
                convolution as the skip connection in the residual network blocks.
            compress (int): Reduced dimensionality in residual branches (from Demucs v3).
            lstm (int): Number of LSTM layers at the end of the encoder.
            trim_right_ratio (float): Ratio for trimming at the right of the transposed
                convolution under the causal setup. If equal to 1.0, it means that all
                the trimming is done at the right.
            res_seq (bool): Whether to use residual sequences.
            last_out_padding (List[Union[int, int]]): Padding for the last output.
            tr_conv_group_ratio (int): Group ratio for transposed convolution.
            conv_group_ratio (int): Group ratio for convolution.

        Examples:
            >>> decoder = SEANetDecoder2d(channels=1, dimension=128, n_filters=32)
            >>> z = torch.randn(1, 32, 256)  # Latent representation
            >>> output = decoder(z)
            >>> print(output.shape)  # Expected output shape: (1, 1, T)

        Note:
            The decoder expects the input tensor to have shape (B, C, T),
            where B is the batch size, C is the number of channels, and T
            is the sequence length.

        Raises:
            AssertionError: If the dimensions of input parameters are inconsistent.
        """
        # [Yihan] changed z in (B, C, T)
        y = self.model(z)
        return y

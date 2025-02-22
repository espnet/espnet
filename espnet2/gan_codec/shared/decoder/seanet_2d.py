# Adapted by Yihan Wu for 2D SEANet (from seanet.py)

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in https://github.com/facebookresearch/encodec/tree/main

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from espnet2.gan_codec.shared.encoder.seanet import (  # noqa
    SLSTM,
    SConv1d,
    apply_parametrization_norm,
    get_extra_padding_for_conv1d,
    get_norm_module,
)
from espnet2.gan_codec.shared.encoder.seanet_2d import SConv2d, get_activation


def unpad2d(x: torch.Tensor, paddings: Tuple[Tuple[int, int], Tuple[int, int]]):
    """Remove padding from x, handling properly zero padding. Only for 1d!"""
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
    """Wrapper around ConvTranspose2d and normalization applied to this conv

    to provide a uniform interface across normalization approaches.
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
        x = self.convtr(x)
        x = self.norm(x)
        return x


class SConvTranspose2d(nn.Module):
    """ConvTranspose2d with some builtin handling of asymmetric or causal padding

    and normalization. Note: causal padding only make sense on time (the last) axis.
    Frequency (the second last) axis are always non-causally padded.
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
    """Residual block from SEANet model.

    Args:
        dim (int): Dimension of the input/output
        kernel_sizes (list): List of kernel sizes for the convolutions.
        dilations (list): List of dilations for the convolutions.
        activation (str): Activation function.
        activation_params (dict): Parameters to provide to the activation function
        norm (str): Normalization method.
        norm_params (dict): Parameters to provide to the underlying normalization
            used along with the convolution.
        causal (bool): Whether to use fully causal convolution.
        pad_mode (str): Padding mode for the convolutions.
        compress (int): Reduced dimensionality in residual branches (from Demucs v3)
        true_skip (bool): Whether to use true skip connection or a simple convolution
            as the skip connection.
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
        return self.shortcut(x) + self.block(
            x
        )  # This is simply the sum of two tensors of the same size


class ReshapeModule(nn.Module):
    def __init__(self, dim=2):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.unsqueeze(x, dim=self.dim)


class SEANetDecoder2d(nn.Module):
    """SEANet decoder.

    Args:
        channels (int): Audio channels.
        dimension (int): Intermediate representation dimension.
        n_filters (int): Base width for the model.
        n_residual_layers (int): nb of residual layers.
        ratios (Sequence[int]): kernel size and stride ratios
        activation (str): Activation function.
        activation_params (dict): Parameters to provide to the activation function
        final_activation (str): Final activation function after all convolutions.
        final_activation_params (dict): Parameters to provide to the activation function
        norm (str): Normalization method.
        norm_params (dict): Parameters to provide to the underlying normalization used
            along with the convolution.
        kernel_size (int): Kernel size for the initial convolution.
        last_kernel_size (int): Kernel size for the initial convolution.
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
        return self.channels

    def forward(self, z):
        # [Yihan] changed z in (B, C, T)
        y = self.model(z)
        return y

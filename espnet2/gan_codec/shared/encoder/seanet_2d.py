# Adapted by Zhihao Du for 2D SEANet (from seanet.py)

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in https://github.com/facebookresearch/encodec/tree/main

"""Encodec SEANet-based encoder and decoder implementation."""

import warnings
from typing import Any, Dict, List, Tuple, Union

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
from espnet2.gan_codec.shared.encoder.snake_activation import Snake1d

if V(torch.__version__) >= V("2.1.0"):
    from torch.nn.utils.parametrizations import weight_norm
else:
    from torch.nn.utils import weight_norm  # noqa


def get_activation(activation: str = None, channels=None, **kwargs):
    if activation.lower() == "snake":
        assert channels is not None, "Snake activation needs channel number."
        return Snake1d(channels=channels)
    else:
        act = getattr(nn, activation)
        return act(**kwargs)


class NormConv2d(nn.Module):
    """Wrapper around Conv2d and normalization applied to this conv

    to provide a uniform interface across normalization approaches.
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
        x = self.conv(x)
        x = self.norm(x)
        return x


def tuple_it(x, num=2):
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
    """Tiny wrapper around F.pad, just to allow for reflect padding on small input.

    If this is the case, we insert extra 0 padding to the right before
    the reflection happen.
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
    """Conv1d with some builtin handling of asymmetric or causal padding

    and normalization. Note: causal padding only make sense on time (the last) axis.
    Frequency (the second last) axis are always non-causally padded.
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
        return self.shortcut(x) + self.block(
            x
        )  # This is simply the sum of two tensors of the same size


class ReshapeModule(nn.Module):
    def __init__(self, dim=2):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.squeeze(x, dim=self.dim)


# Only channels, norm, causal are different between 24HZ & 48HZ,
# everything else is default parameter
# 24HZ -> channels = 1, norm = weight_norm, causal = True
# 48HZ -> channels = 2, norm = time_group_norm, causal = False
class SEANetEncoder2d(nn.Module):
    """SEANet encoder.

    Args:
        input_size (int): Audio channels.
        dimension (int): Intermediate representation dimension.
        n_filters (int): Base width for the model.
        n_residual_layers (int): nb of residual layers.
        ratios (Sequence[int]): kernel size and stride ratios. The encoder uses
            downsampling ratios instead of upsampling ratios, hence it will use
            the ratios in the reverse order to the ones specified here
            that must match the decoder order
        activation (str): Activation function. ELU = Exponential Linear Unit
        activation_params (dict): Parameters to provide to the activation function
        norm (str): Normalization method.
        norm_params (dict): Parameters to provide to the underlying normalization
             used along with the convolution.
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
        if x.dim() == 3:
            x = x.unsqueeze(1)  # x in B,C,T, return B,T,C
        return self.model(x)

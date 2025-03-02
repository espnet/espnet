# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in https://github.com/facebookresearch/encodec/tree/main


import typing as tp
from abc import ABC, abstractmethod

import torch
import torchaudio
from einops import rearrange
from torch import nn

from espnet2.gan_codec.shared.discriminator.msstft_conv import NormConv2d


def get_2d_padding(
    kernel_size: tp.Tuple[int, int], dilation: tp.Tuple[int, int] = (1, 1)
):
    return (
        ((kernel_size[0] - 1) * dilation[0]) // 2,
        ((kernel_size[1] - 1) * dilation[1]) // 2,
    )


class DiscriminatorSTFT(nn.Module):
    """STFT sub-discriminator.

    Args:
        filters (int): Number of filters in convolutions.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        n_fft (int): Size of FFT for each scale.
        hop_length (int): Length of hop between STFT windows for each scale.
        kernel_size (tuple of int): Inner Conv2d kernel sizes.
        stride (tuple of int): Inner Conv2d strides.
        dilations (list of int): Inner Conv2d dilation on the time dimension.
        win_length (int): Window size for each scale.
        normalized (bool): Whether to normalize by magnitude after stft.
        norm (str): Normalization method.
        activation (str): Activation function.
        activation_params (dict): Parameters to provide to the activation function.
        growth (int): Growth factor for the filters.
    """

    def __init__(
        self,
        filters: int,
        in_channels: int = 1,
        out_channels: int = 1,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        max_filters: int = 1024,
        filters_scale: int = 1,
        kernel_size: tp.Tuple[int, int] = (3, 9),
        dilations: tp.List = [1, 2, 4],
        stride: tp.Tuple[int, int] = (1, 2),
        normalized: bool = True,
        norm: str = "weight_norm",
        activation: str = "LeakyReLU",
        activation_params: dict = {"negative_slope": 0.2},
    ):
        super().__init__()
        assert len(kernel_size) == 2
        assert len(stride) == 2
        self.filters = filters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.normalized = normalized
        self.activation = getattr(torch.nn, activation)(**activation_params)
        self.spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window_fn=torch.hann_window,
            normalized=self.normalized,
            center=False,
            pad_mode=None,
            power=None,
        )
        spec_channels = 2 * self.in_channels
        self.convs = nn.ModuleList()
        self.convs.append(
            NormConv2d(
                spec_channels,
                self.filters,
                kernel_size=kernel_size,
                padding=get_2d_padding(kernel_size),
            )
        )
        in_chs = min(filters_scale * self.filters, max_filters)
        for i, dilation in enumerate(dilations):
            out_chs = min((filters_scale ** (i + 1)) * self.filters, max_filters)
            self.convs.append(
                NormConv2d(
                    in_chs,
                    out_chs,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=(dilation, 1),
                    padding=get_2d_padding(kernel_size, (dilation, 1)),
                    norm=norm,
                )
            )
            in_chs = out_chs
        out_chs = min(
            (filters_scale ** (len(dilations) + 1)) * self.filters, max_filters
        )
        self.convs.append(
            NormConv2d(
                in_chs,
                out_chs,
                kernel_size=(kernel_size[0], kernel_size[0]),
                padding=get_2d_padding((kernel_size[0], kernel_size[0])),
                norm=norm,
            )
        )
        self.conv_post = NormConv2d(
            out_chs,
            self.out_channels,
            kernel_size=(kernel_size[0], kernel_size[0]),
            padding=get_2d_padding((kernel_size[0], kernel_size[0])),
            norm=norm,
        )

    def forward(self, x: torch.Tensor):
        fmap = []
        z = self.spec_transform(x)  # [B, 2, Freq, Frames, 2]
        z = torch.cat([z.real, z.imag], dim=1)
        z = rearrange(z, "b c w t -> b c t w")
        for i, layer in enumerate(self.convs):
            z = layer(z)
            z = self.activation(z)
            fmap.append(z)
        z = self.conv_post(z)
        return z, fmap


class MultiDiscriminator(ABC, nn.Module):
    """Base implementation for discriminators composed of sub-discriminators

    acting at different scales.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor):
        raise NotImplementedError("forward method is not implemented")

    @property
    @abstractmethod
    def num_discriminators(self) -> int:
        """Number of discriminators."""
        raise NotImplementedError("num_discriminators is not implemented")


class MultiScaleSTFTDiscriminator(MultiDiscriminator):
    """Multi-Scale STFT (MS-STFT) discriminator.

    Args:
        filters (int): Number of filters in convolutions.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        sep_channels (bool): Separate channels to distinct samples for stereo support.
        n_ffts (Sequence[int]): Size of FFT for each scale.
        hop_lengths (Sequence[int]): Length of hop between STFT windows for each scale.
        win_lengths (Sequence[int]): Window size for each scale.
        **kwargs: Additional args for STFTDiscriminator.
    """

    def __init__(
        self,
        filters: int,
        in_channels: int = 1,
        out_channels: int = 1,
        sep_channels: bool = False,
        n_ffts: tp.List[int] = [1024, 2048, 512],
        hop_lengths: tp.List[int] = [256, 512, 128],
        win_lengths: tp.List[int] = [1024, 2048, 512],
        **kwargs
    ):
        super().__init__()
        assert len(n_ffts) == len(hop_lengths) == len(win_lengths)
        self.sep_channels = sep_channels
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorSTFT(
                    filters,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    n_fft=n_ffts[i],
                    win_length=win_lengths[i],
                    hop_length=hop_lengths[i],
                    **kwargs
                )
                for i in range(len(n_ffts))
            ]
        )

    @property
    def num_discriminators(self):
        return len(self.discriminators)

    def _separate_channels(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        return x.view(-1, 1, T)

    def forward(self, x: torch.Tensor):

        ans = []
        for disc in self.discriminators:
            logit, fmap = disc(x)
            ans.append(fmap + [logit])

        return ans

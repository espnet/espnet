# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""HiFi-GAN Modules.

This code is modified from https://github.com/kan-bayashi/ParallelWaveGAN.

"""

from typing import Any, List

import numpy as np  # noqa
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ModReLU(nn.Module):
    """ComplexReLU module.

    Reference:
        https://arxiv.org/abs/1705.09792
        https://github.com/pytorch/pytorch/issues/47052#issuecomment-718948801
    """

    def __init__(self):
        super().__init__()
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return F.relu(torch.abs(x) + self.b) * torch.exp(1.0j * torch.angle(x))


class ComplexConv2d(nn.Module):
    """ComplexConv2d module."""

    def __init__(self, dim, dim_out, kernel_size, stride=1, padding=0):
        super().__init__()
        conv = nn.Conv2d(dim, dim_out, kernel_size, dtype=torch.complex64)
        self.weight = nn.Parameter(torch.view_as_real(conv.weight))
        self.bias = nn.Parameter(torch.view_as_real(conv.bias))

        self.stride = stride
        self.padding = padding

    def forward(self, x):
        weight, bias = map(torch.view_as_complex, (self.weight, self.bias))

        x = x.to(weight.dtype)
        return F.conv2d(x, weight, bias, stride=self.stride, padding=self.padding)


def ComplexSTFTResidualUnit(in_channel, out_channel, strides):
    """Complex STFT Residual block.

    Args:
        in_channel (int): Input channel.
        out_channel (int): Output channel.
        strides (int): Strides of the whole module.

    Returns:
        nn.Module: Output nn module with complex conv2ds.
    """
    kernel_sizes = tuple(map(lambda t: t + 2, strides))
    paddings = tuple(map(lambda t: t // 2, kernel_sizes))

    return nn.Sequential(
        ComplexConv2d(in_channel, in_channel, 3, padding=1),
        ModReLU(),
        ComplexConv2d(
            in_channel, out_channel, kernel_sizes, stride=strides, padding=paddings
        ),
    )


class ComplexSTFTDiscriminator(nn.Module):
    """ComplexSTFT Discriminator used in SoundStream."""

    def __init__(
        self,
        *,
        in_channels: int = 1,
        channels: int = 32,
        strides: Any = [
            [1, 2],
            [2, 2],
            [1, 2],
            [2, 2],
            [1, 2],
            [2, 2],
        ],
        chan_mults: List[int] = [1, 2, 4, 4, 8, 8],
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        stft_normalized: bool = False,
        logits_abs: bool = True,
    ):
        """Initialize Complex STFT Discriminator used in SoundStream.

        Adapted from https://github.com/alibaba-damo-academy/FunCodec.git

        Args:
            in_channels (int): Input channel.
            channels (int): Output channel.
            strides (List[List(int, int)]): detailed strides in conv2d modules.
            chan_mults (List[int]): Channel multiplers.
            n_fft (int): n_fft in the STFT.
            hop_length (int): hop_length in the STFT.
            stft_normalized (bool): whether to normalize the stft output.
            logits_abs (bool): whether to use the absolute number of output logits.
        """
        super().__init__()
        self.init_conv = ComplexConv2d(in_channels, channels, 7, padding=3)

        layer_channels = tuple(map(lambda mult: mult * channels, chan_mults))
        layer_channels = (channels, *layer_channels)
        layer_channels_pairs = tuple(zip(layer_channels[:-1], layer_channels[1:]))

        self.layers = nn.ModuleList([])

        for layer_stride, (in_channel, out_channel) in zip(
            strides, layer_channels_pairs
        ):
            self.layers.append(
                ComplexSTFTResidualUnit(in_channel, out_channel, layer_stride)
            )

        # stft settings
        self.stft_normalized = stft_normalized
        self.logits_abs = logits_abs

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

    def forward(self, x: torch.Tensor):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input signal (B, 1, T).

        Returns:
            List[List[Tensor]]: List of list of the discriminator output.

        Reference:
            Paper: https://arxiv.org/pdf/2107.03312.pdf
            Implementation: https://github.com/alibaba-damo-academy/FunCodec.git
        """
        x = x.squeeze(1)

        x = torch.stft(
            x,
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            normalized=self.stft_normalized,
            return_complex=True,
        )

        x = rearrange(x, "b ... -> b 1 ...")

        x = self.init_conv(x)

        for layer in self.layers:
            x = layer(x)

        if self.logits_abs:
            x = torch.abs(x)
        else:
            x = torch.view_as_real(x)
        return [[x]]

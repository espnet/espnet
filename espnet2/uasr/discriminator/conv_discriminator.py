import torch
import argparse

import logging
from typing import Optional, Dict
from typeguard import check_argument_types

from espnet2.uasr.discriminator.abs_discriminator import AbsDiscriminator
from espnet2.utils.types import str2bool


class SamePad(torch.nn.Module):
    def __init__(self, kernel_size, causal=False):
        super().__init__()
        if causal:
            self.remove = kernel_size - 1
        else:
            self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x):
        if self.remove > 0:
            x = x[:, :, : -self.remove]
        return x


class ConvDiscriminator(AbsDiscriminator):
    """convolutional discriminator for UASR."""

    def __init__(
        self,
        input_dim: int,
        cfg: Optional[Dict] = None,
        conv_channels: int = 384,
        conv_kernel: int = 8,
        conv_dilation: int = 1,
        conv_depth: int = 2,
        linear_emb: str2bool = False,
        causal: str2bool = True,
        max_pool: str2bool = False,
        act_after_linear: str2bool = False,
        dropout: float = 0.0,
        spectral_norm: str2bool = False,
        weight_norm: str2bool = False,
    ):
        super().__init__()
        assert check_argument_types()
        if cfg is not None:
            cfg = argparse.Namespace(**cfg)
            self.conv_channels = cfg.discriminator_dim
            self.conv_kernel = cfg.discriminator_kernel
            self.conv_dilation = cfg.discriminator_dilation
            self.conv_depth = cfg.discriminator_depth
            self.linear_emb = cfg.discriminator_linear_emb
            self.causal = cfg.discriminator_causal
            self.max_pool = cfg.discriminator_max_pool
            self.act_after_linear = cfg.discriminator_act_after_linear
            self.dropout = cfg.discriminator_dropout
            self.spectral_norm = cfg.discriminator_spectral_norm
            self.weight_norm = cfg.discriminator_weight_norm
        else:
            self.conv_channels = conv_channels
            self.conv_kernel = conv_kernel
            self.conv_dilation = conv_dilation
            self.conv_depth = conv_depth
            self.linear_emb = linear_emb
            self.causal = causal
            self.max_pool = max_pool
            self.act_after_linear = act_after_linear
            self.dropout = dropout
            self.spectral_norm = spectral_norm
            self.weight_norm = weight_norm

        if self.causal:
            self.conv_padding = self.conv_kernel - 1
        else:
            self.conv_padding = self.conv_kernel // 2

        def make_conv(
            in_channel, out_channel, kernal_size, padding_size=0, dilation_value=1
        ):
            conv = torch.nn.Conv1d(
                in_channel,
                out_channel,
                kernel_size=kernal_size,
                padding=padding_size,
                dilation=dilation_value,
            )
            if self.spectral_norm:
                conv = torch.nn.utils.spectral_norm(conv)
            elif self.weight_norm:
                conv = torch.nn.utils.weight_norm(conv)
            return conv

        # initialize embedding
        if self.linear_emb:
            emb_net = [
                make_conv(
                    input_dim, self.conv_channels, 1, dilation_value=self.conv_dilation
                )
            ]
        else:
            emb_net = [
                make_conv(
                    input_dim,
                    self.conv_channels,
                    self.conv_kernel,
                    self.conv_padding,
                    dilation_value=self.conv_dilation,
                ),
                SamePad(kernel_size=self.conv_kernel, causal=self.causal),
            ]

        if self.act_after_linear:
            emb_net.append(torch.nn.GELU())

        # initialize inner conv
        inner_net = [
            torch.nn.Sequential(
                make_conv(
                    self.conv_channels,
                    self.conv_channels,
                    self.conv_kernel,
                    self.conv_padding,
                    dilation_value=self.conv_dilation,
                ),
                SamePad(kernel_size=self.conv_kernel, causal=self.causal),
                torch.nn.Dropout(self.dropout),
                torch.nn.GELU(),
            )
            for _ in range(self.conv_depth - 1)
        ]

        inner_net += [
            make_conv(
                self.conv_channels,
                1,
                self.conv_kernel,
                self.conv_padding,
                dilation_value=1,
            ),
            SamePad(kernel_size=self.conv_kernel, causal=self.causal),
        ]

        self.net = torch.nn.Sequential(
            *emb_net,
            torch.nn.Dropout(dropout),
            *inner_net,
        )

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor]):
        assert check_argument_types()

        # (Batch, Time, Channel) -> (Batch, Channel, Time)
        x = x.transpose(1, 2)

        x = self.net(x)

        # (Batch, Channel, Time) -> (Batch, Time, Channel)
        x = x.transpose(1, 2)
        x_sz = x.size(1)
        if padding_mask is not None and padding_mask.any() and padding_mask.dim() > 1:
            padding_mask = padding_mask[:, : x.size(1)]
            padding_mask.to(x.device)
            x[padding_mask] = float("-inf") if self.max_pool else 0
            x_sz = x_sz - padding_mask.sum(dim=-1)

        x = x.squeeze(-1)
        if self.max_pool:
            x, _ = x.max(dim=-1)
        else:
            x = x.sum(dim=-1)
            x = x / x_sz

        return x

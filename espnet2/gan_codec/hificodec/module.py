import math  # noqa
from dataclasses import dataclass
from typing import Optional

import numpy as np  # noqa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm, weight_norm

from espnet2.gan_codec.shared.quantizer.residual_vq import ResidualVectorQuantizer

LRELU_SLOPE = 0.1


@dataclass
class QuantizedResult:
    quantized: torch.Tensor
    codes: torch.Tensor
    bandwidth: torch.Tensor  # bandwidth in kb/s used, per batch item.
    penalty: Optional[torch.Tensor] = None


class Generator(torch.nn.Module):
    def __init__(
        self,
        upsample_rates,
        upsample_kernel_sizes,
        upsample_initial_channel,
        resblock_num,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        out_dim,
    ):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = weight_norm(
            Conv1d(out_dim, upsample_initial_channel, 7, 1, padding=3)
        )
        resblock = ResBlock1 if resblock_num == "1" else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        # padding=(u//2 + u%2),
                        padding=(k - u) // 2,
                        # output_padding=u%2
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x, LRELU_SLOPE)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print("Removing weight norm...")
        for layers in self.ups:
            remove_weight_norm(layers)
        for layers in self.resblocks:
            layers.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class Encoder(torch.nn.Module):
    def __init__(
        self,
        resblock_num,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_kernel_sizes,
    ):
        super(Encoder, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = weight_norm(Conv1d(1, 32, 7, 1, padding=3))
        self.normalize = nn.ModuleList()
        resblock = ResBlock1 if resblock_num == "1" else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(
            list(reversed(list(zip(upsample_rates, upsample_kernel_sizes))))
        ):
            self.ups.append(
                weight_norm(
                    Conv1d(
                        32 * (2**i),
                        32 * (2 ** (i + 1)),
                        k,
                        u,
                        padding=((k - u) // 2),
                        # padding=(u//2 + u%2)
                    )
                )
            )
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = 32 * (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(
                    list(reversed(resblock_kernel_sizes)),
                    list(reversed(resblock_dilation_sizes)),
                )
            ):
                self.resblocks.append(resblock(ch, k, d))
                self.normalize.append(
                    torch.nn.GroupNorm(ch // 16, ch, eps=1e-6, affine=True)
                )
        self.conv_post = Conv1d(512, 512, 3, 1, padding=1)
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                    xs = self.normalize[i * self.num_kernels + j](xs)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
                    xs = self.normalize[i * self.num_kernels + j](xs)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        return x

    def remove_weight_norm(self):
        print("Removing weight norm...")
        for layers in self.ups:
            remove_weight_norm(layers)
        for layers in self.resblocks:
            layers.remove_weight_norm()
        remove_weight_norm(self.conv_pre)


class GroupResidualVectorQuantization(nn.Module):
    def __init__(
        self,
        quantizer_target_bandwidth,
        hidden_dim,
        quantizer_n_q,
        quantizer_bins,
        quantizer_decay,
        quantizer_kmeans_init,
        quantizer_kmeans_iters,
        quantizer_threshold_ema_dead_code,
        **kwargs
    ):
        super().__init__()

        self.quantizer1 = ResidualVectorQuantizer(
            dimension=hidden_dim,
            n_q=quantizer_n_q,
            bins=quantizer_bins,
            decay=quantizer_decay,
            kmeans_init=quantizer_kmeans_init,
            kmeans_iters=quantizer_kmeans_iters,
            threshold_ema_dead_code=quantizer_threshold_ema_dead_code,
        )
        self.quantizer0 = ResidualVectorQuantizer(
            dimension=hidden_dim,
            n_q=quantizer_n_q,
            bins=quantizer_bins,
            decay=quantizer_decay,
            kmeans_init=quantizer_kmeans_init,
            kmeans_iters=quantizer_kmeans_iters,
            threshold_ema_dead_code=quantizer_threshold_ema_dead_code,
        )

        self.l1_quantization_loss = torch.nn.L1Loss(reduction="mean")
        self.l2_quantization_loss = torch.nn.MSELoss(reduction="mean")

        self.target_bandwidths = quantizer_target_bandwidth

    def forward(
        self, xin: torch.Tensor, sample_rate: int, bandwidth: Optional[float] = None
    ) -> QuantizedResult:
        # x: [B,T,D]

        xin = xin.transpose(1, 2)
        # x = xin.reshape(-1, 512)
        x = xin
        # x = torch.split(x, 512 // self.h.n_code_groups, dim=-1)

        x0, x1 = torch.split(x, 512 // 2, dim=-1)
        x0 = x0.transpose(1, 2)
        x1 = x1.transpose(1, 2)

        if bandwidth is None:
            bw = self.target_bandwidths[-1]
        else:
            bw = bandwidth

        quantized1, _, _, commit_loss1 = self.quantizer1(x1, sample_rate, bw)

        quantized0, _, _, commit_loss0 = self.quantizer0(x0, sample_rate, bw)

        quantized = torch.cat([quantized0, quantized1], dim=1)

        commit_loss = commit_loss0 + commit_loss1

        quantization_loss1 = self.l1_quantization_loss(
            x1, quantized1.detach()
        ) + self.l2_quantization_loss(x1, quantized1.detach())

        quantization_loss0 = self.l1_quantization_loss(
            x0, quantized0.detach()
        ) + self.l2_quantization_loss(x0, quantized0.detach())

        quantization_loss = quantization_loss0 + quantization_loss1

        return quantized, _, _, commit_loss, quantization_loss

    def encode(
        self,
        xin: torch.Tensor,
        frame_rate: int,
        target_bw: Optional[float] = None,
    ):
        """HiFICodec codec encoding.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 1, T).
        Returns:
            torch.Tensor: neural codecs in shape ().
        """

        x = xin

        x0, x1 = torch.split(x, 512 // 2, dim=1)

        if target_bw is None:
            bw = self.target_bandwidths[-1]
        else:
            bw = target_bw

        codes0 = self.quantizer0.encode(x0, frame_rate, bw)
        codes1 = self.quantizer1.encode(x1, frame_rate, bw)
        code = torch.cat([codes0, codes1], dim=1)

        return code

    def decode(self, code: torch.Tensor):
        """HiFICodec codec decoding.

        Args:
            codes (torch.Tensor): neural codecs in shape ().
        Returns:
            torch.Tensor: resynthesized audio.
        """

        code0, code1 = torch.split(code, 2 // 2, dim=1)

        quantized0 = self.quantizer0.decode(code0)
        quantized1 = self.quantizer1.decode(code1)
        quantized = torch.cat([quantized0, quantized1], dim=1)

        return quantized


class ResBlock1(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
            ]
        )
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for layers in self.convs1:
            remove_weight_norm(layers)
        for layers in self.convs2:
            remove_weight_norm(layers)


class ResBlock2(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.convs = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
            ]
        )
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for layers in self.convs:
            remove_weight_norm(layers)


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)

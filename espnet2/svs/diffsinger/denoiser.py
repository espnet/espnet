import math
from math import sqrt
from typing import Optional, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Mish(nn.Module):
    """Mish Activation Function.
    Introduced in `Mish: A Self Regularized Non-Monotonic Activation Function`_.
    .. _Mish: A Self Regularized Non-Monotonic Activation Function:
       https://arxiv.org/abs/1908.08681
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate forward propagation.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor.
        """
        return x * torch.tanh(F.softplus(x))


class SinusoidalPosEmb(nn.Module):
    """
    Diffusion step emebedding
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ResidualBlock(nn.Module):
    """Residual Block for Diffusion Denoiser."""

    def __init__(
        self,
        adim: int,
        channels: int,
        dilation: int,
    ) -> None:
        """Initialization.
        Args:
            adim (int): Size of dimensions.
            channels (int): Number of channels.
            dilation (int): Size of dilations.
        """
        super().__init__()
        self.conv = nn.Conv1d(
            channels, 2 * channels, 3, padding=dilation, dilation=dilation
        )
        self.diff_proj = nn.Linear(channels, channels)
        self.cond_proj = nn.Conv1d(adim, 2 * channels, 1)
        self.out_proj = nn.Conv1d(channels, 2 * channels, 1)

    def forward(
        self, x: torch.Tensor, condition: torch.Tensor, step: torch.Tensor
    ) -> Union[torch.Tensor, torch.Tensor]:
        """Calculate forward propagation.
        Args:
            x (torch.Tensor): Input tensor.
            condition (torch.Tensor): Conditioning tensor.
            step (torch.Tensor): Number of diffusion step.
        Returns:
            Union[torch.Tensor, torch.Tensor]: Output tensor.
        """
        step = self.diff_proj(step).unsqueeze(-1)
        condition = self.cond_proj(condition)
        y = x + step
        y = self.conv(y) + condition
        gate, _filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(_filter)
        y = self.out_proj(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / math.sqrt(2.0), skip


def Conv1d(*args, **kwargs):
    layer = nn.Conv1d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class DiffNet(nn.Module):
    """
    Denoiser network (Wavenet based Denoiser network)
    Introduced in `Diffsinger: Singing Voice Synthesis via Shallow Diffusion Mechanism`
    .. _Diffsinger: Singing Voice Synthesis via Shallow Diffusion Mechanism:
       https://arxiv.org/abs/2105.02446

    """

    def __init__(
        self,
        encoder_hidden: int,
        residual_layers: int,
        residual_channels: int,
        dilation_cycle_length: int,
        in_dims: int = 80,
    ):
        """
        Args:
            encoder_hidden: hiddden channel
            residual_layers: residual block number
            residual_channels:
        """
        super().__init__()
        self.input_projection = Conv1d(in_dims, residual_channels, 1)
        self.diffusion_embedding = SinusoidalPosEmb(residual_channels)
        dim = residual_channels
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), Mish(), nn.Linear(dim * 4, dim)
        )
        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    encoder_hidden, residual_channels, 2 ** (i % dilation_cycle_length)
                )
                for i in range(residual_layers)
            ]
        )
        self.skip_projection = Conv1d(residual_channels, residual_channels, 1)
        self.output_projection = Conv1d(residual_channels, in_dims, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(
        self,
        spec: torch.Tensor,
        diffusion_step: torch.Tensor,
        cond: torch.Tensor,
    ):
        """
        Args:
            spec: mel-spectrum [B, 1, M, T]
            diffusion_step: diffusion step [B, 1]
            cond: music condition information [B, M, T]
        return:
            x: random noise [B, 1, M/80, T] (same size with spec)

        """
        x = spec[:, 0]  # [B, 1, M]
        x = self.input_projection(x)  # x [B, residual_channel, T]
        x = F.relu(x)

        diffusion_step = self.diffusion_embedding(diffusion_step)
        diffusion_step = self.mlp(diffusion_step)

        skip = []
        for layer_id, layer in enumerate(self.residual_layers):
            x, skip_connection = layer(x, cond, diffusion_step)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)  # [B, 80, T]
        return x[:, None, :, :]  # [B, 1, 80, T]

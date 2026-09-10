"""Conditional flow-matching vocoder for Sidon: SSL features -> 48 kHz.

The generative alternative to the GAN vocoders. A velocity field
v(x_t, t | c) is trained with conditional flow matching on straight paths
between Gaussian noise x_0 and the clean waveform x_1 (Lipman et al. 2023;
Liu et al. 2023): x_t = (1 - t) x_0 + t x_1 with target velocity x_1 - x_0.
No discriminator, no adversarial or feature-matching loss. Synthesis
integrates dx/dt = v(x, t | c) from t = 0 to 1 with a midpoint solver in a
small number of steps (``num_steps``, default 16), so inference cost is
``num_steps`` x 2 velocity evaluations per utterance.

Network: ESPnet's non-causal WaveNet (gan_tts/wavenet) at the waveform rate
with dilated gated residual blocks, as in DiffWave / PriorGrad. A DAC-style
transposed-convolution stack lifts the 50 Hz features to the 48 kHz
conditioning it takes as local input; a sinusoidal embedding of t is its
global conditioning.
"""

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
from torch.nn.utils import remove_weight_norm, weight_norm

from espnet2.enh.decoder.sidon_vocoder import DecoderBlock
from espnet2.gan_tts.wavenet import WaveNet


class TimestepEmbedding(nn.Module):
    """Sinusoidal embedding of t in [0, 1] followed by a small MLP."""

    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device) / half
        )
        ang = t[:, None].float() * 1000.0 * freqs[None]
        emb = torch.cat([ang.sin(), ang.cos()], dim=-1)
        return self.mlp(emb)


class FeatureUpsampler(nn.Module):
    """50 Hz SSL features -> conditioning at the waveform rate (DAC blocks)."""

    def __init__(
        self,
        input_dim: int,
        channels: int = 256,
        rates: Sequence[int] = (8, 5, 4, 3, 2),
        out_channels: int = 64,
    ):
        super().__init__()
        layers = [weight_norm(nn.Conv1d(input_dim, channels, kernel_size=7, padding=3))]
        dim = channels
        for i, stride in enumerate(rates):
            out = max(out_channels, channels // 2 ** (i + 1))
            layers.append(DecoderBlock(dim, out, stride))
            dim = out
        layers.append(
            weight_norm(nn.Conv1d(dim, out_channels, kernel_size=7, padding=3))
        )
        self.net = nn.Sequential(*layers)
        self.upsample_factor = int(math.prod(rates))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """(B, T, D) -> (B, C, T * upsample_factor)."""
        return self.net(feat.transpose(1, 2))


class SidonFlowVocoder(nn.Module):
    """Velocity network with feature conditioning; ``generate`` samples audio.

    Args:
        input_dim: SSL feature dimension.
        cond_channels: channels of the waveform-rate conditioning.
        cond_hidden: width of the first upsampler layer (halved per stage).
        rates: upsampling strides, product must equal output_sr / 50.
        layers, stacks, residual_channels, gate_channels, skip_channels,
            kernel_size: WaveNet geometry (dilation 2**i within each stack).
        time_channels: width of the timestep embedding.
        num_steps: midpoint ODE steps used by ``generate`` by default.
        data_scale: gain applied to the waveform before it is matched
            against unit-variance noise (speech sits at RMS 0.01-0.1;
            without it the velocity target is nearly all noise removal).
            ``generate`` divides it back out.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        cond_channels: int = 64,
        cond_hidden: int = 256,
        rates: Sequence[int] = (8, 5, 4, 3, 2),
        layers: int = 30,
        stacks: int = 3,
        residual_channels: int = 64,
        gate_channels: int = 128,
        skip_channels: int = 64,
        kernel_size: int = 3,
        time_channels: int = 128,
        num_steps: int = 16,
        data_scale: float = 10.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.data_scale = data_scale
        self.upsampler = FeatureUpsampler(input_dim, cond_hidden, rates, cond_channels)
        self.upsample_factor = self.upsampler.upsample_factor
        self.time_embed = TimestepEmbedding(time_channels, time_channels)
        self.net = WaveNet(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            layers=layers,
            stacks=stacks,
            base_dilation=2,
            residual_channels=residual_channels,
            aux_channels=cond_channels,
            gate_channels=gate_channels,
            skip_channels=skip_channels,
            global_channels=time_channels,
            use_first_conv=True,
            use_last_conv=True,
            scale_skip_connect=True,
        )
        # Zero output at initialisation (v = 0, so the loss starts at the
        # variance of the target), the usual choice for diffusion nets. The
        # conv is weight-normalised: zero the magnitude, not the direction.
        last = [m for m in self.net.last_conv.modules() if hasattr(m, "weight_g")][-1]
        nn.init.zeros_(last.weight_g)
        nn.init.zeros_(last.bias)
        self.num_steps = num_steps

    def condition(self, ssl_feat: torch.Tensor) -> torch.Tensor:
        """(B, T, D) features -> (B, cond_channels, T * 960)."""
        return self.upsampler(ssl_feat)

    def velocity(
        self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        """x_t (B, 1, T), t (B,), cond (B, C, T) -> v (B, 1, T)."""
        n = min(x_t.size(-1), cond.size(-1))
        g = self.time_embed(t).unsqueeze(-1)
        return self.net(x_t[..., :n], c=cond[..., :n], g=g)

    @torch.no_grad()
    def generate(
        self, ssl_feat: torch.Tensor, num_steps: Optional[int] = None
    ) -> torch.Tensor:
        """(B, T, D) features -> (B, T * 960) waveform by midpoint ODE steps."""
        steps = num_steps or self.num_steps
        cond = self.condition(ssl_feat)
        x = torch.randn(
            cond.size(0), 1, cond.size(-1), device=cond.device, dtype=cond.dtype
        )
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((x.size(0),), i * dt, device=x.device)
            k1 = self.velocity(x, t, cond)
            k2 = self.velocity(x + 0.5 * dt * k1, t + 0.5 * dt, cond)
            x = x + dt * k2
        return (x.squeeze(1) / self.data_scale).clamp(-1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, D, T_frames) -> (B, 1, T): the GAN vocoders' calling convention."""
        return self.generate(x.transpose(1, 2)).unsqueeze(1)

    def remove_weight_norm(self) -> None:
        """Fold weight norm into plain weights for inference."""
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
                try:
                    remove_weight_norm(module)
                except ValueError:
                    pass  # already folded

# Implemented from
# (https://github.com/Rongjiehuang/ProDiff/blob/main/modules/ProDiff/model/ProDiff_teacher.py)
# Copyright 2022 Hitachi LTD. (Nelson Yalta)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
import math
from typing import Optional, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding


def _vpsde_beta_t(t: int, T: int, min_beta: float, max_beta: float) -> float:
    """Beta Scheduler.

    Args:
        t (int): current step.
        T (int): total steps.
        min_beta (float): minimum beta.
        max_beta (float): maximum beta.

    Returns:
        float: current beta.

    """
    t_coef = (2 * t - 1) / (T**2)
    return 1.0 - np.exp(-min_beta / T - 0.5 * (max_beta - min_beta) * t_coef)


def noise_scheduler(
    sched_type: str,
    timesteps: int,
    min_beta: float = 0.0,
    max_beta: float = 0.01,
    s: float = 0.008,
) -> torch.Tensor:
    """Noise Scheduler.

    Args:
        sched_type (str): type of scheduler.
        timesteps (int): numbern of time steps.
        min_beta (float, optional): Minimum beta. Defaults to 0.0.
        max_beta (float, optional): Maximum beta. Defaults to 0.01.
        s (float, optional): Scheduler intersection. Defaults to 0.008.

    Returns:
        tensor: Noise.

    """
    if sched_type == "linear":
        scheduler = np.linspace(1e-6, 0.01, timesteps)

    elif sched_type == "cosine":
        steps = timesteps + 1
        x = np.linspace(0, steps, steps)
        alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        scheduler = np.clip(betas, a_min=0, a_max=0.999)

    elif sched_type == "vpsde":
        scheduler = np.array(
            [
                _vpsde_beta_t(t, timesteps, min_beta, max_beta)
                for t in range(1, timesteps + 1)
            ]
        )
    else:
        raise NotImplementedError

    return torch.as_tensor(scheduler.astype(np.float32))


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


class SpectogramDenoiser(nn.Module):
    """Spectogram Denoiser.

    Ref: https://arxiv.org/pdf/2207.06389.pdf.
    """

    def __init__(
        self,
        idim: int,
        adim: int = 256,
        layers: int = 20,
        channels: int = 256,
        cycle_length: int = 1,
        timesteps: int = 200,
        timescale: int = 1,
        max_beta: float = 40.0,
        scheduler: str = "vpsde",
        dropout_rate: float = 0.05,
    ) -> None:
        """Initialization.

        Args:
            idim (int): Dimension of the inputs.
            adim (int, optional):Dimension of the hidden states. Defaults to 256.
            layers (int, optional): Number of layers. Defaults to 20.
            channels (int, optional): Number of channels of each layer. Defaults to 256.
            cycle_length (int, optional): Cycle length of the diffusion. Defaults to 1.
            timesteps (int, optional): Number of timesteps of the diffusion.
                Defaults to 200.
            timescale (int, optional): Number of timescale. Defaults to 1.
            max_beta (float, optional): Maximum beta value for schedueler.
                Defaults to 40.
            scheduler (str, optional): Type of noise scheduler. Defaults to "vpsde".
            dropout_rate (float, optional): Dropout rate. Defaults to 0.05.

        """
        super().__init__()
        self.idim = idim
        self.timesteps = timesteps
        self.scale = timescale
        self.num_layers = layers
        self.channels = channels

        # Denoiser
        self.in_proj = nn.Conv1d(idim, channels, 1)
        self.denoiser_pos = PositionalEncoding(channels, dropout_rate)
        self.denoiser_mlp = nn.Sequential(
            nn.Linear(channels, channels * 4), Mish(), nn.Linear(channels * 4, channels)
        )
        self.denoiser_res = nn.ModuleList(
            [
                ResidualBlock(adim, channels, 2 ** (i % cycle_length))
                for i in range(layers)
            ]
        )
        self.skip_proj = nn.Conv1d(channels, channels, 1)
        self.feats_out = nn.Conv1d(channels, idim, 1)

        # Diffusion
        self.betas = noise_scheduler(scheduler, timesteps + 1, 0.1, max_beta, 8e-3)
        alphas = 1.0 - self.betas
        alphas_cumulative = torch.cumprod(alphas, dim=0)
        self.register_buffer("alphas_cumulative", torch.sqrt(alphas_cumulative))
        self.register_buffer(
            "min_alphas_cumulative", torch.sqrt(1.0 - alphas_cumulative)
        )

    def forward(
        self,
        xs: torch.Tensor,
        ys: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
        is_inference: bool = False,
    ) -> torch.Tensor:
        """Calculate forward propagation.

        Args:
            xs (torch.Tensor): Phoneme-encoded tensor (#batch, time, dims)
            ys (Optional[torch.Tensor], optional): Mel-based reference
                tensor (#batch, time, mels). Defaults to None.
            masks (Optional[torch.Tensor], optional): Mask tensor (#batch, time).
                Defaults to None.

        Returns:
            torch.Tensor: Output tensor (#batch, time, dims).

        """
        if is_inference:
            return self.inference(xs)

        batch_size = xs.shape[0]
        timesteps = (
            torch.randint(0, self.timesteps + 1, (batch_size,)).to(xs.device).long()
        )

        # Diffusion
        ys_noise = self.diffusion(ys, timesteps)  # (batch, 1, dims, time)
        ys_noise = ys_noise * masks.unsqueeze(1)

        # Denoise
        ys_denoise = self.forward_denoise(ys_noise, timesteps, xs)
        ys_denoise = ys_denoise * masks
        return ys_denoise.transpose(1, 2)

    def forward_denoise(
        self, xs_noisy: torch.Tensor, step: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        """Calculate forward for denoising diffusion.

        Args:
            xs_noisy (torch.Tensor): Input tensor.
            step (torch.Tensor): Number of step.
            condition (torch.Tensor): Conditioning tensor.

        Returns:
            torch.Tensor: Denoised tensor.

        """
        xs_noisy = xs_noisy.squeeze(1)
        condition = condition.transpose(1, 2)
        xs_noisy = F.relu(self.in_proj(xs_noisy))

        step = step.unsqueeze(-1).expand(-1, self.channels)
        step = self.denoiser_pos(step.unsqueeze(1)).squeeze(1)
        step = self.denoiser_mlp(step)

        skip_conns = list()
        for _, layer in enumerate(self.denoiser_res):
            xs_noisy, skip = layer(xs_noisy, condition, step)
            skip_conns.append(skip)

        xs_noisy = torch.sum(torch.stack(skip_conns), dim=0) / math.sqrt(
            self.num_layers
        )
        xs_denoise = F.relu(self.skip_proj(xs_noisy))
        xs_denoise = self.feats_out(xs_noisy)
        return xs_denoise

    def diffusion(
        self,
        xs_ref: torch.Tensor,
        steps: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Calculate diffusion process during training.

        Args:
            xs_ref (torch.Tensor): Input tensor.
            steps (torch.Tensor): Number of step.
            noise (Optional[torch.Tensor], optional): Noise tensor. Defaults to None.

        Returns:
            torch.Tensor: Output tensor.

        """
        # here goes norm_spec if does something
        batch_size = xs_ref.shape[0]
        xs_ref = xs_ref.transpose(1, 2).unsqueeze(1)
        steps = torch.clamp(steps, min=0)  # not sure if this is required

        # make a noise tensor
        if noise is None:
            noise = torch.randn_like(xs_ref)

        # q-sample
        ndims = (batch_size, *((1,) * (xs_ref.dim() - 1)))
        cum_prods = self.alphas_cumulative.gather(-1, steps).reshape(ndims)
        min_cum_prods = self.min_alphas_cumulative.gather(-1, steps).reshape(ndims)
        xs_noisy = xs_ref * cum_prods + noise * min_cum_prods
        return xs_noisy

    def inference(self, condition: torch.Tensor) -> torch.Tensor:
        """Calculate forward during inference.

        Args:
            condition (torch.Tensor): Conditioning tensor (batch, time, dims).

        Returns:
            torch.Tensor: Output tensor.

        """
        batch = condition.shape[0]
        device = condition.device
        shape = (batch, 1, self.idim, condition.shape[1])
        xs_noisy = torch.randn(shape).to(device)  # (batch, 1, dims, time)

        # required params:
        beta = self.betas
        alph = 1.0 - beta
        alph_prod = torch.cumprod(alph, axis=0)
        alph_prod_prv = torch.cat((torch.ones((1,)), alph_prod[:-1]))
        coef1 = beta * torch.sqrt(alph_prod_prv) / (1.0 - alph_prod)
        coef2 = (1.0 - alph_prod_prv) * torch.sqrt(alph) / (1.0 - alph_prod)
        post_var = beta * (1.0 - alph_prod_prv) / (1.0 - alph_prod)
        post_var = torch.log(torch.maximum(post_var, torch.full((1,), 1e-20)))

        # allows non CPU denoising
        coef1 = coef1.to(device)
        coef2 = coef2.to(device)
        post_var = post_var.to(device)

        # denoising steps
        for _step in reversed(range(0, self.timesteps)):
            # p-sample
            steps = torch.full((batch,), _step, dtype=torch.long).to(device)
            xs_denoised = self.forward_denoise(xs_noisy, steps, condition).unsqueeze(1)

            # q-posterior (xs_denoised, xs_noisy, steps)
            ndims = (batch, *((1,) * (xs_denoised.dim() - 1)))
            _coef1 = coef1.gather(-1, steps).reshape(ndims)
            _coef2 = coef2.gather(-1, steps).reshape(ndims)
            q_mean = _coef1 * xs_denoised + _coef2 * xs_noisy
            q_log_var = post_var.gather(-1, steps).reshape(ndims)

            # q-posterior-sample
            noise = torch.randn_like(xs_denoised).to(device)
            _mask = (1 - (steps == 0).float()).reshape(ndims)
            xs_noisy = q_mean + _mask * (0.5 * q_log_var).exp() * noise
        ys = xs_noisy[0].transpose(1, 2)
        return ys

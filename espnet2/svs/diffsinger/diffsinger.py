"""Diffsinger related modules."""

import logging

from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Tuple

from espnet2.svs.abs_svs import AbsSVS
from espnet2.svs.diffsinger.fftsinger import FFTSinger
from espnet2.svs.diffsinger.denoiser import DiffNet
from espnet2.svs.diffsinger.diffLoss import DiffLoss

import torch
import torch.nn.functional as F
from espnet2.torch_utils.initialize import initialize

import random
import numpy as np
from functools import partial

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1))) 
    # [b, ] -> [b, 1, 1, ..., 1]

def noise_like(shape, device, repeat=False):
    """
    repeat_noise (1, ...) * shape[0] on dim=0
    """
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


def cosine_beta_schedule(timesteps, s=0.008):
    """
    step embedding: cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    scheduler = np.clip(betas, a_min=0, a_max=0.999)
    return torch.as_tensor(scheduler.astype(np.float32))


class DiffSinger(AbsSVS):

    def __init__(
        self,
        idim: int,
        odim: int,
        midi_dim: int = 129,
        tempo_dim: int = 500,
        embed_dim: int = 256, 
        adim: int = 384,
        aheads: int = 4,
        elayers: int = 6, 
        eunits: int = 1536, 
        dlayers: int = 6, 
        dunits: int = 1536, 
        postnet_layers: int = 5, 
        postnet_chans: int = 512, 
        use_batch_norm: bool = True, 
        reduction_factor: int = 1, 
        encoder_type: str = "transformer",
        decoder_type: str = "transformer",
        denoiser_type: str = 'wavenet',
        feats_minmax: Optional[Dict[str, torch.Tensor]] = None,
        #diffnet
        diffnet_encoder_hidden: int = 256,
        diffnet_residual_layers: int = 20,
        diffnet_residual_channels: int = 256,
        diffnet_dilation_cycle_length: int = 4,
        diffnet_input_dim: int = 80,
        # other 
        K_step: int = 1000,
        mel_bins: int = 80, 
        timesteps: int = 1000,
        shallow_diffusion: bool = False,
        init_type: str = "xavier_uniform",
        use_masking: bool = False,
        use_weighted_masking: bool = False,
        loss_type: str = 'l1',
    ) -> None:
        super().__init__()

        self.K_step = K_step
        self.shallow_diffusion = shallow_diffusion
        self.mel_bins = mel_bins
        self.num_timesteps = timesteps
        self.loss_type = loss_type

        self.fftsinger = FFTSinger(
            idim,
            odim,
            midi_dim=midi_dim,
            tempo_dim=tempo_dim,
            embed_dim=embed_dim,
            adim=adim,
            aheads=aheads,
            elayers=elayers, 
            eunits=eunits, 
            dlayers=dlayers, 
            dunits=dunits, 
            postnet_layers=postnet_layers, 
            postnet_chans=postnet_chans, 
            use_batch_norm=use_batch_norm, 
            reduction_factor=reduction_factor, 
            init_type=init_type,
            use_masking=use_masking, 
            loss_type=loss_type, 
            encoder_type=encoder_type, 
            decoder_type=decoder_type, 
        )

        if denoiser_type == 'wavenet':
            self.denoiser = DiffNet(
                diffnet_encoder_hidden,
                diffnet_residual_layers,
                diffnet_residual_channels,
                diffnet_dilation_cycle_length,
                diffnet_input_dim,
            )
        else:
            raise ValueError(f"{denoiser_type} is not supported.") 
        
        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat((torch.tensor[1.], alphas_cumprod[:-1]), 0)

        timesteps = betas.size(0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to =1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', posterior_variance)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', torch.log(torch.maximum(posterior_variance, 1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        self.register_buffer('spec_min', feats_minmax['feats_min'][None, None, :mel_bins])
        self.register_buffer('spec_max', feats_minmax['feats_max'][None, None, :mel_bins])

        # initialize parameters
        self._reset_parameters(
            init_type=init_type,
        )

        self.criterion = DiffLoss(
            use_masking=use_masking, use_weighted_masking=use_weighted_masking
        )


    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        feats: torch.Tensor,
        feats_lengths: torch.Tensor,
        label: Optional[Dict[str, torch.Tensor]] = None,
        label_lengths: Optional[Dict[str, torch.Tensor]] = None,
        melody: Optional[Dict[str, torch.Tensor]] = None,
        melody_lengths: Optional[Dict[str, torch.Tensor]] = None,
        pitch: Optional[torch.Tensor] = None,
        pitch_lengths: Optional[torch.Tensor] = None,
        duration: Optional[Dict[str, torch.Tensor]] = None,
        duration_lengths: Optional[Dict[str, torch.Tensor]] = None,
        spembs: Optional[torch.Tensor] = None,
        sids: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
        joint_training: bool = False,
        flag_IsValid=False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate forward propagation.

        Args:
            text (LongTensor): Batch of padded character ids (B, T_text).
            text_lengths (LongTensor): Batch of lengths of each input (B,).
            feats (Tensor): Batch of padded target features (B, T_feats, odim).
            feats_lengths (LongTensor): Batch of the lengths of each target (B,).
            label (Optional[Dict]): key is "lab" or "score";
                value (LongTensor): Batch of padded label ids (B, Tmax).
            label_lengths (Optional[Dict]): key is "lab" or "score";
                value (LongTensor): Batch of the lengths of padded label ids (B, ).
            melody (Optional[Dict]): key is "lab" or "score";
                value (LongTensor): Batch of padded melody (B, Tmax).
            melody_lengths (Optional[Dict]): key is "lab" or "score";
                value (LongTensor): Batch of the lengths of padded melody (B, ).
            pitch (FloatTensor): Batch of padded f0 (B, Tmax).
            pitch_lengths (LongTensor): Batch of the lengths of padded f0 (B, ).
            duration (Optional[Dict]): key is "lab", "score_phn" or "score_syb";
                value (LongTensor): Batch of padded duration (B, Tmax).
            duration_length (Optional[Dict]): key is "lab", "score_phn" or "score_syb";
                value (LongTensor): Batch of the lengths of padded duration (B, ).
            spembs (Optional[Tensor]): Batch of speaker embeddings (B, spk_embed_dim).
            sids (Optional[Tensor]): Batch of speaker IDs (B, 1).
            lids (Optional[Tensor]): Batch of language IDs (B, 1).
            joint_training (bool): Whether to perform joint training with vocoder.

        Returns:
            Tensor: Loss scalar value.
            Dict: Statistics to be monitored.
            Tensor: Weight value if not joint training else model outputs.
        """

        hs = self.fftsinger(
            text,
            text_lengths,
            feats,
            feats_lengths,
            label,
            label_lengths,
            melody,
            melody_lengths,
            pitch,
            pitch_lengths,
            duration,
            duration_lengths,
            spembs,
            sids,
            lids,
            skip_decoder = True,
            joint_training = False,
        )
        
        text = text[:, : text_lengths.max()]  # for data-parallel
        mel_gt = feats[:, : feats_lengths.max()]  # for data-parallel
        
        batch_size, device = text.size(0), text.device
        cond = hs.transpose(1, 2) # cond: [B, adim, T]
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device).long()
        x = self.norm_spec(mel_gt)
        x = x.transpose(1, 2)[:, None, :, :] # x: [B, 1, M, T]
        
        noise, noise_pred, mel_pred = self.predict_noise(x, t, cond)

    

    def inference(
        self,
        text: torch.Tensor,
        feats: Optional[torch.Tensor] = None,
        label: Optional[Dict[str, torch.Tensor]] = None,
        melody: Optional[Dict[str, torch.Tensor]] = None,
        pitch: Optional[torch.Tensor] = None,
        duration: Optional[Dict[str, torch.Tensor]] = None,
        spembs: Optional[torch.Tensor] = None,
        sids: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
        joint_training: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Generate the sequence of features given the sequences of characters.

        Args:
            text (LongTensor): Input sequence of characters (T_text,).
            feats (Optional[Tensor]): Feature sequence to extract style (N, idim).
            durations (Optional[LongTensor]): Groundtruth of duration (T_text + 1,).
            label (Optional[Dict]): key is "lab" or "score";
                value (LongTensor): Batch of padded label ids (Tmax).
            melody (Optional[Dict]): key is "lab" or "score";
                value (LongTensor): Batch of padded melody (Tmax).
            pitch (FloatTensor): Batch of padded f0 (B, Tmax).
            duration (Optional[Dict]): key is "lab", "score_phn" or "score_syb";
                value (LongTensor): Batch of padded duration (Tmax).
            spembs (Optional[Tensor]): Speaker embedding (spk_embed_dim,).
            sids (Optional[Tensor]): Speaker ID (1,).
            lids (Optional[Tensor]): Language ID (1,).
            alpha (float): Alpha to control the speed.

        Returns:
            Dict[str, Tensor]: Output dict including the following items:
                * feat_gen (Tensor): Output sequence of features (T_feats, odim).
                * duration (Tensor): Duration sequence (T_text + 1,).
        """
        hs, fs2_mel = self.fftsinger.inference(
            text,
            feats,
            label,
            melody,
            pitch,
            duration,
            spembs,
            sids,
            lids,
            joint_training = False,
        )
        
        batch_size, device = text.size(0), text.device
        cond = hs.transpose(1, 2) # cond: [B, adim, T]

        t = self.K_step
        fs2_mel = self.norm_spec(fs2_mel)
        fs2_mel = fs2_mel.transpose(1, 2)[:, None, :, :] # [B, 1, T, M]

        x = self.q_sample(x_start=fs2_mel, t=torch.tensor([t - 1], device=device).long())
        if self.shallow_diffusion is False:
            shape = (cond.shape[0], 1, self.mel_bins, cond.shape[2]) # [B, 1, out_dims, T]
            x = torch.randn(shape, device=device)
        for i in reversed(range(0, t)):
            x = self.p_sample(x, torch.full((batch_size, ), i, device=device, dtype=torch.long), cond)
        x = x[:, 0].transpose(1, 2)
        mel_out = self.denorm_spec(x)
        return dict(
            feat_gen=mel_out, prob=None, att_w=None
        ) # outs, probs, att_ws

    
    def predict_start_from_noise(self, x_t, t, noise):
        """
        predict mel_start(without noise) from noisy mel
        Args:
            x_t: noise mel
            t: diffusion step
            noise: pred_noise
        return:
            predict mel
        """
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    
    def q_posterior(self, x_start, x_t, t):
        """
        calculations for posterior q(x_{t-1} | x_t, x_0)        
        """
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        ) # mu(x_t, x_0)
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped


    def p_mean_variance(self, x_t, t, cond, clip_denoised: bool):
        """
        calc mean and variance of posterior q(x_{t-1} | x_t, x_0)

        Args:
            x_t: noise mel
            t: diffusion step
            cond: music score
            clip_denoised: limit min and max
        Return:
            mean and variance of posterior q(x_{t-1} | x_t, x_0)
        """
        noise_pred = self.denoiser(x_t, t, cond=cond)
        x_recon = self.predict_start_from_noise(x_t, t=t, noise=noise_pred)
        if clip_denoised:
            x_recon.clamp_(-1., 1.)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x_t, t=t)
        return model_mean, posterior_variance, posterior_log_variance


    @torch.no_grad()
    def p_sample(self, x_t, t, cond, clip_denoised=True, repeat_noise=False):
        """
        predict model and resample noise

        Args:
            x_t: noise mel
            t: diffusion step
            cond: music score
        Return:
            Resample noise
        """
        b, *_, device = *x_t.shape, x_t.device
        model_mean, _, model_log_variance = self.p_mean_variance(x_t=x_t, t=t, cond=cond, clip_denoised=clip_denoised)
        noise = noise_like(x_t.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x_t.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise


    def q_sample(self, x_start, t, noise=None):
        """
        diffusion process:
        Args:
            x_start: mel
            t: diffusio step
        Return:
            noise mel
        """
        if noise is None: 
            noise = torch.randn_like(x_start)
        return(
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )


    def predict_noise(self, x_start, t, cond, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)
        noise_pred = self.denoiser(x_t, t, cond)
        mel_pred = self.predict_start_from_noise(x_t, t=t, noise=noise_pred)

        return noise, noise_pred, mel_pred  


    def norm_spec(self, x):
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1

    def denorm_spec(self, x):
        return (x + 1) / 2 * (self.spec_max - self.spec_min) + self.spec_min

    def _reset_parameters(
        self, init_type: str,
    ):
        # initialize parameters
        if init_type != "pytorch":
            initialize(self, init_type)

"""Diffsinger related modules."""

import logging
import random
from functools import partial
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from espnet2.svs.abs_svs import AbsSVS
from espnet2.svs.diffsinger.denoiser import DiffNet
from espnet2.svs.diffsinger.diffLoss import DiffLoss
from espnet2.svs.diffsinger.fftsinger import FFTSinger
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.torch_utils.initialize import initialize
from espnet.nets.pytorch_backend.tacotron2.decoder import Postnet


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))
    # [b, ] -> [b, 1, 1, ..., 1]


def noise_like(shape, device, repeat=False):
    """
    repeat_noise (1, ...) * shape[0] on dim=0
    """
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device).repeat(
        shape[0], *((1,) * (len(shape) - 1))
    )
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


def linear_beta_schedule(timesteps, max_beta):
    """
    linear schedule
    """
    betas = np.linspace(1e-4, max_beta, timesteps)
    return betas


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
        postnet_filts: int = 5,
        postnet_dropout_rate: float = 0.5,
        use_batch_norm: bool = True,
        reduction_factor: int = 1,
        encoder_type: str = "transformer",
        decoder_type: str = "transformer",
        denoiser_type: str = "wavenet",
        feats_minmax: Optional[Dict[str, torch.Tensor]] = None,
        # diffnet
        diffnet_encoder_hidden: int = 256,
        diffnet_residual_layers: int = 20,
        diffnet_residual_channels: int = 256,
        diffnet_dilation_cycle_length: int = 4,
        diffnet_input_dim: int = 80,
        # extra embedding related
        spks: Optional[int] = None,
        langs: Optional[int] = None,
        spk_embed_dim: Optional[int] = None,
        spk_embed_integration_type: str = "add",
        # other
        schedule_type: str = "linear",
        max_beta: float = 0.01,
        K_step: int = 1000,
        mel_bins: int = 80,
        timesteps: int = 1000,
        shallow_diffusion: bool = True,
        init_type: str = "xavier_uniform",
        use_masking: bool = False,
        use_weighted_masking: bool = False,
        loss_type: str = "L1",
    ) -> None:
        super().__init__()

        self.K_step = K_step
        self.shallow_diffusion = shallow_diffusion
        self.mel_bins = mel_bins
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
        for k, v in self.fftsinger.named_parameters():
            v.requires_grad = False

        if denoiser_type == "wavenet":
            self.denoiser = DiffNet(
                diffnet_encoder_hidden,
                diffnet_residual_layers,
                diffnet_residual_channels,
                diffnet_dilation_cycle_length,
                diffnet_input_dim,
            )
        else:
            raise ValueError(f"{denoiser_type} is not supported.")

        if schedule_type == "linear":
            betas = cosine_beta_schedule(timesteps)
        else:
            betas = linear_beta_schedule(timesteps, max_beta)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat((torch.tensor([1.0]), alphas_cumprod[:-1]), 0)

        timesteps = betas.size(0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        # above: equal to =1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer("posterior_variance", posterior_variance)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.max(posterior_variance, torch.tensor([1e-20]))),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        if feats_minmax is not None:
            self.register_buffer(
                "spec_min", feats_minmax["feats_min"][None, None, :mel_bins]
            )
            self.register_buffer(
                "spec_max", feats_minmax["feats_max"][None, None, :mel_bins]
            )

        # define spk and lang embedding
        self.spks = None
        if spks is not None and spks > 1:
            self.spks = spks
            self.sid_emb = torch.nn.Embedding(spks, adim)
        self.langs = None
        if langs is not None and langs > 1:
            self.langs = langs
            self.lid_emb = torch.nn.Embedding(langs, adim)

        # define additional projection for speaker embedding
        self.spk_embed_dim = None
        if spk_embed_dim is not None and spk_embed_dim > 0:
            self.spk_embed_dim = spk_embed_dim
            self.spk_embed_integration_type = spk_embed_integration_type
        if self.spk_embed_dim is not None:
            if self.spk_embed_integration_type == "add":
                self.projection = torch.nn.Linear(self.spk_embed_dim, adim)
            else:
                self.projection = torch.nn.Linear(adim + self.spk_embed_dim, adim)

        self.postnet = None
        # # define postnet
        # self.postnet = (
        #     None
        #     if postnet_layers == 0
        #     else Postnet(
        #         idim=idim,
        #         odim=odim,
        #         n_layers=postnet_layers,
        #         n_chans=postnet_chans,
        #         n_filts=postnet_filts,
        #         use_batch_norm=use_batch_norm,
        #         dropout_rate=postnet_dropout_rate,
        #     )
        # )

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
        feats_minmax: Optional[Dict[str, torch.Tensor]] = None,
        label: Optional[Dict[str, torch.Tensor]] = None,
        label_lengths: Optional[Dict[str, torch.Tensor]] = None,
        melody: Optional[Dict[str, torch.Tensor]] = None,
        melody_lengths: Optional[Dict[str, torch.Tensor]] = None,
        pitch: Optional[torch.Tensor] = None,
        pitch_lengths: Optional[torch.Tensor] = None,
        duration: Optional[Dict[str, torch.Tensor]] = None,
        duration_lengths: Optional[Dict[str, torch.Tensor]] = None,
        slur: torch.LongTensor = None,
        slur_lengths: torch.Tensor = None,
        spembs: Optional[torch.Tensor] = None,
        sids: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
        joint_training: bool = False,
        flag_IsValid=False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
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

        hs, d_outs = self.fftsinger(
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
            skip_decoder=True,
            joint_training=False,
        )

        if joint_training:
            label_lengths = label_lengths
            ds = duration
        else:
            label_lengths = label_lengths["score"]
            ds = duration["lab"]

        text = text[:, : text_lengths.max()]  # for data-parallel
        mel_gt = feats[:, : feats_lengths.max()]  # for data-parallel

        batch_size, device = text.size(0), text.device
        # [B, T, adim] -> [B, adim, T]
        cond = hs.transpose(1, 2)
        t = torch.randint(0, self.K_step, (batch_size,), device=device).long()
        x = self.norm_spec(mel_gt, feats_minmax)
        # [B, T, M] -> [B, 1, M, T]
        x = x.transpose(1, 2)[:, None, :, :]

        noise, noise_pred, mel_pred = self.predict_noise(x, t, cond)

        # noise_mask = feats.transpose(1, 2) != 0.0
        noise_mask = None
        noise_l1_loss, noise_l2_loss, duration_loss = self.criterion(
            noise, noise_pred, noise_mask, d_outs, ds, label_lengths, self.loss_type
        )
        loss = noise_l1_loss + noise_l2_loss + duration_loss
        if self.loss_type == 'L1':
            stats = dict(
                loss=loss.item(),
                noise_l1_loss=noise_l1_loss.item(),
                duration_loss=duration_loss.item(),
            )
        else:
            stats = dict(
                loss=loss.item(),
                noise_l2_loss=noise_l2_loss.item(),
                duration_loss=duration_loss.item(),
            )
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)

        return loss, stats, weight

    def inference(
        self,
        text: torch.Tensor,
        feats: Optional[torch.Tensor] = None,
        feats_minmax: Optional[Dict[str, torch.Tensor]] = None,
        label: Optional[Dict[str, torch.Tensor]] = None,
        melody: Optional[Dict[str, torch.Tensor]] = None,
        pitch: Optional[torch.Tensor] = None,
        duration: Optional[Dict[str, torch.Tensor]] = None,
        slur: Optional[Dict[str, torch.Tensor]] = None,
        spembs: Optional[torch.Tensor] = None,
        sids: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
        joint_training: bool = False,
        use_teacher_forcing: bool = False,
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
            joint_training=False,
        )  # hs:[B, T, adim], fs2_mel:[B, T, M]

        batch_size, device = text.size(0), text.device
        # [B, T, adim] -> [B, adim, T]
        cond = hs.transpose(1, 2)

        t = self.K_step

        shape = (cond.shape[0], 1, self.mel_bins, cond.shape[2])  # [B, 1, M, T]
        x = torch.randn(shape, device=device)  # naive diffsinger

        if self.shallow_diffusion is True:  # diffsinger with shallow diffusion
            fs2_mel = self.norm_spec(fs2_mel, feats_minmax)
            fs2_mel = fs2_mel.transpose(1, 2)[:, None, :, :]  # [B, 1, M, T]
            x = self.q_sample(
                x_start=fs2_mel, t=torch.tensor([t - 1], device=device).long()
            )

        for i in tqdm(reversed(range(0, t)), desc="sample time step", total=t):
            x = self.p_sample(
                x, torch.full((batch_size,), i, device=device, dtype=torch.long), cond
            )

        # [B, 1, M, T] -> [B, M, T] -> [B, T, M] and B = 1
        x = x[:, 0].transpose(1, 2)
        before_outs = self.denorm_spec(x, feats_minmax)

        if self.postnet is None:
            after_outs = before_outs
        else:
            after_outs = before_outs + self.postnet(
                before_outs.transpose(1, 2)
            ).transpose(1, 2)

        return dict(
            feat_gen=after_outs[0], prob=None, att_w=None
        )  # outs, probs, att_ws

    def q_mean_variance(self, x_start, t):
        """
        mean and variance of data distrubution q(x)
        """
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

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
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        """
        calculations for posterior q(x_{t-1} | x_t, x_0)
        q(x_{t-1} | x_t, x_0) is a Guassian distrubution

        Return:
            mean and variance of posterior q(x_{t-1} | x_t, x_0)
        """
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )  # mu(x_t, x_0)
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
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
            x_recon.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x_t, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x_t, t, cond, clip_denoised=True, repeat_noise=False):
        """
        denoise process
        Args:
            x_t: noise mel
            t: diffusion step
            cond: music score
        Return:
            mel after one denoise step
        """
        b, *_, device = *x_t.shape, x_t.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x_t=x_t, t=t, cond=cond, clip_denoised=clip_denoised
        )
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
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def predict_noise(self, x_start, t, cond, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)
        noise_pred = self.denoiser(x_t, t, cond)
        mel_pred = self.predict_start_from_noise(x_t, t=t, noise=noise_pred)

        return noise, noise_pred, mel_pred

    def norm_spec(self, x, feats_minmax):
        spec_min = feats_minmax['feats_min']
        spec_max = feats_minmax['feats_max']
        return (x - spec_min) / (spec_max - spec_min) * 2 - 1

    def denorm_spec(self, x, feats_minmax):
        spec_min = feats_minmax['feats_min'].unsqueeze(0)
        spec_max = feats_minmax['feats_max'].unsqueeze(0)
        return (x + 1) / 2 * (spec_max - spec_min) + spec_min

    def _reset_parameters(
        self,
        init_type: str,
    ):
        # initialize parameters
        if init_type != "pytorch":
            initialize(self, init_type)

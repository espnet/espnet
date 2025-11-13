# Copyright 2025 Haoran Wang
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import functools
import logging
import math
import random
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from typeguard import typechecked

from espnet2.gan_codec.abs_gan_codec import AbsGANCodec
from espnet2.gan_codec.dac.dac import DACDiscriminator
from espnet2.gan_codec.shared.decoder.seanet import SEANetDecoder
from espnet2.gan_codec.shared.encoder.seanet import SEANetEncoder
from espnet2.gan_codec.shared.loss.freq_loss import MultiScaleMelSpectrogramLoss
from espnet2.gan_codec.shared.quantizer.band_vq import BandSimVQ, BandVQ
from espnet2.gan_tts.hifigan.loss import (
    DiscriminatorAdversarialLoss,
    FeatureMatchLoss,
    GeneratorAdversarialLoss,
)
from espnet2.torch_utils.device_funcs import force_gatherable


class BSCodec(AbsGANCodec):
    """Band Codec with Band SimVQ model."""

    @typechecked
    def __init__(
        self,
        sampling_rate: int = 24000,
        generator_params: Dict[str, Any] = {
            "hidden_dim": 128,
            "bands": [
                (0, 500),
                (500, 2000),
                (2000, 12000),
            ],
            "encdec_channels": 1,
            "encdec_n_filters": 32,
            "encdec_n_residual_layers": 1,
            "encdec_ratios": [8, 5, 4, 2],
            "encdec_activation": "Snake",
            "encdec_activation_params": {},
            "encdec_norm": "weight_norm",
            "encdec_norm_params": {},
            "encdec_kernel_size": 7,
            "encdec_residual_kernel_size": 7,
            "encdec_last_kernel_size": 7,
            "encdec_dilation_base": 2,
            "encdec_causal": False,
            "encdec_pad_mode": "reflect",
            "encdec_true_skip": False,
            "encdec_compress": 2,
            "encdec_lstm": 2,
            "quantize_choice": "band_vq",  # options: band_vq, band_simvq
            "quantize_codebook_size": 1024,
            "quantize_codebook_dim": 128,
            "decoder_trim_right_ratio": 1.0,
            "decoder_final_activation": None,
            "decoder_final_activation_params": None,
        },
        discriminator_params: Dict[str, Any] = {
            "scale_follow_official_norm": False,
            "msmpmb_discriminator_params": {
                "rates": [],
                "fft_sizes": [2048, 1024, 512],
                "sample_rate": 24000,
                "periods": [2, 3, 5, 7, 11],
                "period_discriminator_params": {
                    "in_channels": 1,
                    "out_channels": 1,
                    "kernel_sizes": [5, 3],
                    "channels": 32,
                    "downsample_scales": [3, 3, 3, 3, 1],
                    "max_downsample_channels": 1024,
                    "bias": True,
                    "nonlinear_activation": "LeakyReLU",
                    "nonlinear_activation_params": {"negative_slope": 0.1},
                    "use_weight_norm": True,
                    "use_spectral_norm": False,
                },
                "band_discriminator_params": {
                    "hop_factor": 0.25,
                    "sample_rate": 24000,
                    "bands": [
                        (0.0, 0.1),
                        (0.1, 0.25),
                        (0.25, 0.5),
                        (0.5, 0.75),
                        (0.75, 1.0),
                    ],
                    "channel": 32,
                },
            },
        },
        # loss related
        generator_adv_loss_params: Dict[str, Any] = {
            "average_by_discriminators": False,
            "loss_type": "mse",
        },
        discriminator_adv_loss_params: Dict[str, Any] = {
            "average_by_discriminators": False,
            "loss_type": "mse",
        },
        use_feat_match_loss: bool = True,
        feat_match_loss_params: Dict[str, Any] = {
            "average_by_discriminators": False,
            "average_by_layers": False,
            "include_final_outputs": True,
        },
        use_mel_loss: bool = True,
        mel_loss_params: Dict[str, Any] = {
            "fs": 24000,
            "range_start": 6,
            "range_end": 11,
            "window": "hann",
            "n_mels": 80,
            "fmin": 0,
            "fmax": None,
            "log_base": None,
        },
        use_dual_decoder: bool = True,
        lambda_quantization: float = 1.0,
        lambda_reconstruct: float = 1.0,
        lambda_adv: float = 1.0,
        lambda_feat_match: float = 2.0,
        lambda_mel: float = 45.0,
        cache_generator_outputs: bool = False,
    ):
        super().__init__()
        self.num_bands = len(generator_params["bands"])
        # define modules
        generator_params.update(sample_rate=sampling_rate)
        self.generator = BSCodecGenerator(**generator_params)
        self.discriminator = DACDiscriminator(**discriminator_params)
        self.generator_adv_loss = GeneratorAdversarialLoss(
            **generator_adv_loss_params,
        )
        self.generator_reconstruct_loss = torch.nn.L1Loss(reduction="mean")
        self.discriminator_adv_loss = DiscriminatorAdversarialLoss(
            **discriminator_adv_loss_params,
        )
        self.use_feat_match_loss = use_feat_match_loss
        if self.use_feat_match_loss:
            self.feat_match_loss = FeatureMatchLoss(
                **feat_match_loss_params,
            )
        self.use_mel_loss = use_mel_loss
        mel_loss_params.update(fs=sampling_rate)
        if self.use_mel_loss:
            self.mel_loss = MultiScaleMelSpectrogramLoss(
                **mel_loss_params,
            )
        self.use_dual_decoder = use_dual_decoder
        if self.use_dual_decoder:
            assert self.use_mel_loss, "only use dual decoder with Mel loss"

        # coefficients
        self.lambda_reconstruct = lambda_reconstruct
        self.lambda_quantization = lambda_quantization
        self.lambda_adv = lambda_adv
        if self.use_feat_match_loss:
            self.lambda_feat_match = lambda_feat_match
        if self.use_mel_loss:
            self.lambda_mel = lambda_mel

        # cache
        self.cache_generator_outputs = cache_generator_outputs
        self._cache = None

        # store sampling rate for saving wav file
        # (not used for the training)
        self.fs = sampling_rate
        self.frame_shift = functools.reduce(
            lambda x, y: x * y, generator_params["encdec_ratios"]
        )
        self.codebook_size = generator_params["quantize_codebook_size"]
        self.code_size_per_stream = None

    def meta_info(self) -> Dict[str, Any]:
        return {
            "fs": self.fs,
            "num_streams": self.num_bands,
            "frame_shift": self.frame_shift,
            "codebook_size": self.codebook_size,
            "code_size_per_stream": self.code_size_per_stream,
        }

    def forward(
        self,
        audio: torch.Tensor,
        forward_generator: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """Perform generator forward.

        Args:
            audio (Tensor): Audio waveform tensor (B, T_wav).
            forward_generator (bool): Whether to forward generator.

        Returns:
            Dict[str, Any]:
                - loss (Tensor): Loss scalar tensor.
                - stats (Dict[str, float]): Statistics to be monitored.
                - weight (Tensor): Weight tensor to summarize losses.
                - optim_idx (int): Optimizer index (0 for G and 1 for D).

        """
        if forward_generator:
            return self._forward_generator(
                audio=audio,
                **kwargs,
            )
        else:
            return self._forward_discriminator(
                audio=audio,
                **kwargs,
            )

    def _forward_generator(
        self,
        audio: torch.Tensor,
        **kwargs,
    ) -> Dict[str, Any]:
        """Perform generator forward.

        Args:
            audio (Tensor): Audio waveform tensor (B, T_wav).

        Returns:
            Dict[str, Any]:
                - loss (Tensor): Loss scalar tensor.
                - stats (Dict[str, float]): Statistics to be monitored.
                - weight (Tensor): Weight tensor to summarize losses.
                - optim_idx (int): Optimizer index (0 for G and 1 for D).

        """
        # setup
        batch_size = audio.size(0)

        audio = audio.unsqueeze(1)

        # calculate generator outputs
        reuse_cache = True
        if not self.cache_generator_outputs or self._cache is None:
            reuse_cache = False
            audio_hat = self.generator(audio, use_dual_decoder=self.use_dual_decoder)
        else:
            audio_hat = self._cache

        # store cache
        if self.training and self.cache_generator_outputs and not reuse_cache:
            self._cache = audio_hat

        # Use full reconstructed audio for discriminator
        p_hat = self.discriminator(audio_hat[0])
        with torch.no_grad():
            # do not store discriminator gradient in generator turn
            p = self.discriminator(audio)

        # calculate losses
        adv_loss = self.generator_adv_loss(p_hat)
        adv_loss = adv_loss * self.lambda_adv
        vq_loss = audio_hat[3] * self.lambda_quantization

        subband_reconstruct_loss = 0.0
        for i in range(self.num_bands):
            subband_reconstruct_loss += self.generator_reconstruct_loss(
                audio_hat[1][:, i, :].unsqueeze(1), audio_hat[2][:, i, :].unsqueeze(1)
            )
        subband_reconstruct_loss = (
            subband_reconstruct_loss * self.lambda_reconstruct
        ) / self.num_bands
        reconstruct_loss = (
            self.generator_reconstruct_loss(audio, audio_hat[0])
            * self.lambda_reconstruct
        )
        loss = adv_loss + vq_loss + subband_reconstruct_loss + reconstruct_loss
        stats = dict(
            adv_loss=adv_loss.item(),
            vq_loss=vq_loss.item(),
            subband_reconstruct_loss=subband_reconstruct_loss.item(),
            reconstruct_loss=reconstruct_loss.item(),
        )

        if self.use_feat_match_loss:
            feat_match_loss = self.feat_match_loss(p_hat, p)
            feat_match_loss = feat_match_loss * self.lambda_feat_match
            loss = loss + feat_match_loss
            stats.update(feat_match_loss=feat_match_loss.item())
        if self.use_mel_loss:
            subband_mel_loss = 0.0
            for i in range(self.num_bands):
                subband_mel_loss += self.mel_loss(
                    audio_hat[2][:, i, :].unsqueeze(1),
                    audio_hat[1][:, i, :].unsqueeze(1),
                )
            subband_mel_loss = (subband_mel_loss * self.lambda_mel) / self.num_bands
            mel_loss = self.mel_loss(audio_hat[0], audio)
            mel_loss = self.lambda_mel * mel_loss
            loss = loss + subband_mel_loss + mel_loss
            stats.update(
                mel_loss=mel_loss.item(), subband_mel_loss=subband_mel_loss.item()
            )
            if self.use_dual_decoder:
                mel_loss_real = self.mel_loss(audio_hat[4], audio)
                mel_loss_real = self.lambda_mel * mel_loss_real
                loss = loss + mel_loss_real
                stats.update(mel_loss_real=mel_loss_real.item())

        stats.update(loss=loss.item())

        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)

        # reset cache
        if reuse_cache or not self.training:
            self._cache = None

        return {
            "loss": loss,
            "stats": stats,
            "weight": weight,
            "optim_idx": 0,  # needed for trainer
        }

    def _forward_discriminator(
        self,
        audio: torch.Tensor,
        **kwargs,
    ) -> Dict[str, Any]:
        """Perform generator forward.

        Args:
            audio (Tensor): Audio waveform tensor (B, T_wav).

        Returns:
            Dict[str, Any]:
                - loss (Tensor): Loss scalar tensor.
                - stats (Dict[str, float]): Statistics to be monitored.
                - weight (Tensor): Weight tensor to summarize losses.
                - optim_idx (int): Optimizer index (0 for G and 1 for D).

        """

        # setup
        batch_size = audio.size(0)
        audio = audio.unsqueeze(1)

        # calculate generator outputs
        reuse_cache = True
        if not self.cache_generator_outputs or self._cache is None:
            reuse_cache = False
            audio_hat = self.generator(audio)
        else:
            audio_hat = self._cache

        # store cache
        if self.cache_generator_outputs and not reuse_cache:
            self._cache = (audio_hat,)

        # Use full reconstructed audio for discriminator
        p_hat = self.discriminator(audio_hat[0].detach())
        p = self.discriminator(audio)

        # calculate losses
        real_loss, fake_loss = self.discriminator_adv_loss(p_hat, p)
        loss = real_loss + fake_loss

        stats = dict(
            discriminator_loss=loss.item(),
            real_loss=real_loss.item(),
            fake_loss=fake_loss.item(),
        )
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)

        # reset cache
        if reuse_cache or not self.training:
            self._cache = None

        return {
            "loss": loss,
            "stats": stats,
            "weight": weight,
            "optim_idx": 1,  # needed for trainer
        }

    def inference(
        self,
        x: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Run inference.

        Args:
            x (Tensor): Input audio (T_wav,).

        Returns:
            Dict[str, Tensor]:
                - wav (Tensor): Generated waveform tensor (T_wav,).
                - codec (Tensor): Generated neural codec (T_code, N_stream).

        """
        codec = self.generator.encode(x)
        wav = self.generator.decode(codec)

        return {"wav": wav, "codec": codec}

    def encode(
        self,
        x: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Run encoding.

        Args:
            x (Tensor): Input audio (T_wav,).

        Returns:
            Tensor: Generated codes (T_code, N_stream).

        """
        return self.generator.encode(x)

    def decode(
        self,
        x: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Run encoding.

        Args:
            x (Tensor): Input codes (T_code, N_stream).

        Returns:
            Tensor: Generated waveform (T_wav,).

        """
        return self.generator.decode(x)


class BSCodecGenerator(nn.Module):
    """Band Similarity RVQ generator module."""

    @typechecked
    def __init__(
        self,
        sample_rate: int = 24000,
        bands: List[Any] = [
            (0, 500),
            (500, 2000),
            (2000, 12000),
        ],
        hidden_dim: int = 128,
        encdec_channels: int = 1,
        encdec_n_filters: int = 32,
        encdec_n_residual_layers: int = 1,
        encdec_ratios: List[int] = [8, 5, 4, 2],
        encdec_activation: str = "Snake",
        encdec_activation_params: Dict[str, Any] = {},
        encdec_norm: str = "weight_norm",
        encdec_norm_params: Dict[str, Any] = {},
        encdec_kernel_size: int = 7,
        encdec_residual_kernel_size: int = 7,
        encdec_last_kernel_size: int = 7,
        encdec_dilation_base: int = 2,
        encdec_causal: bool = False,
        encdec_pad_mode: str = "reflect",
        encdec_true_skip: bool = False,
        encdec_compress: int = 2,
        encdec_lstm: int = 2,
        quantize_choice: str = "band_vq",
        quantize_codebook_size: int = 1024,
        quantize_codebook_dim: int = 128,
        decoder_trim_right_ratio: float = 1.0,
        decoder_final_activation: Optional[str] = None,
        decoder_final_activation_params: Optional[dict] = None,
    ):
        super().__init__()

        # Initialize encoder
        self.sample_rate = sample_rate
        self.bands = bands
        self.num_bands = len(bands)
        self.frame_rate = math.ceil(sample_rate / np.prod(encdec_ratios))

        self.encoders = nn.ModuleList(
            [
                SEANetEncoder(
                    channels=encdec_channels,
                    dimension=hidden_dim,
                    n_filters=encdec_n_filters,
                    n_residual_layers=encdec_n_residual_layers,
                    ratios=encdec_ratios,
                    activation=encdec_activation,
                    activation_params=encdec_activation_params,
                    norm=encdec_norm,
                    norm_params=encdec_norm_params,
                    kernel_size=encdec_kernel_size,
                    residual_kernel_size=encdec_residual_kernel_size,
                    last_kernel_size=encdec_last_kernel_size,
                    dilation_base=encdec_dilation_base,
                    causal=encdec_causal,
                    pad_mode=encdec_pad_mode,
                    true_skip=encdec_true_skip,
                    compress=encdec_compress,
                    lstm=encdec_lstm,
                )
                for _ in bands
            ]
        )

        if quantize_choice == "band_vq":
            self.quantizer = BandVQ(
                num_bands=len(self.bands),
                dimension=hidden_dim,
                bins=quantize_codebook_size,
                codebook_dim=quantize_codebook_dim,
            )
        elif quantize_choice == "band_simvq":
            self.quantizer = BandSimVQ(
                num_bands=len(self.bands),
                dim=hidden_dim,
                codebook_size=quantize_codebook_size,
                codebook_dim=quantize_codebook_dim,
            )
        else:
            raise ValueError(
                f"Unknown quantize_choice: {quantize_choice}. Options: band_vq, band_simvq"
            )

        # Initialize decoder
        self.decoders = nn.ModuleList(
            [
                SEANetDecoder(
                    channels=encdec_channels,
                    dimension=hidden_dim,
                    n_filters=encdec_n_filters,
                    n_residual_layers=encdec_n_residual_layers,
                    ratios=encdec_ratios,
                    activation=encdec_activation,
                    activation_params=encdec_activation_params,
                    norm=encdec_norm,
                    norm_params=encdec_norm_params,
                    kernel_size=encdec_kernel_size,
                    residual_kernel_size=encdec_residual_kernel_size,
                    last_kernel_size=encdec_last_kernel_size,
                    dilation_base=encdec_dilation_base,
                    causal=encdec_causal,
                    pad_mode=encdec_pad_mode,
                    true_skip=encdec_true_skip,
                    compress=encdec_compress,
                    lstm=encdec_lstm,
                    trim_right_ratio=decoder_trim_right_ratio,
                    final_activation=decoder_final_activation,
                    final_activation_params=decoder_final_activation_params,
                )
                for _ in bands
            ]
        )

    def forward(self, x: torch.Tensor, use_dual_decoder: bool = False):
        """BSCodec forward propagation.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 1, T).
            use_dual_decoder (bool): Whether to use dual decoder.
        Returns:
            tuple: (y_rec, subbands, resyn_audio_all, vq_loss, resyn_audio_real)
        """
        y = x.squeeze(1)
        subbands = split_audio_bands(y, self.sample_rate, self.bands)

        encoder_out = []
        for i, enc in enumerate(self.encoders):
            xi = subbands[:, i, :].unsqueeze(1)
            hi = enc(xi)
            encoder_out.append(hi)
        encoder_out = torch.stack(encoder_out, dim=1)

        quantized, codes, vq_loss = self.quantizer(encoder_out)
        resyn_audio_all = []
        for i, dec in enumerate(self.decoders):
            resyn_audio_all.append(dec(quantized[:, i, :, :]).squeeze(1))
        resyn_audio_all = torch.stack(resyn_audio_all, dim=1)
        y_rec = reconstruct_audio_bands(resyn_audio_all)
        if use_dual_decoder:
            resyn_audio_real_all = []
            for i, dec in enumerate(self.decoders):
                resyn_audio_real_all.append(dec(encoder_out[:, i, :, :]).squeeze(1))
            resyn_audio_real_all = torch.stack(resyn_audio_real_all, dim=1)
            resyn_audio_real = reconstruct_audio_bands(resyn_audio_real_all).unsqueeze(
                1
            )
        else:
            resyn_audio_real = None
        return y_rec.unsqueeze(1), subbands, resyn_audio_all, vq_loss, resyn_audio_real

    def encode(
        self,
        x: torch.Tensor,
    ):
        """BSCodec codec encoding.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 1, T).
        Returns:
            torch.Tensor: neural codecs in shape (B, N_bands, T_frames).
        """

        y = x.squeeze(1)
        subbands = split_audio_bands(y, self.sample_rate, self.bands)

        encoder_out = []
        for i, enc in enumerate(self.encoders):
            xi = subbands[:, i, :].unsqueeze(1)
            hi = enc(xi)
            encoder_out.append(hi)
        encoder_out = torch.stack(encoder_out, dim=1)
        codes_all = self.quantizer.encode(encoder_out)
        return codes_all

    def decode(self, codes: torch.Tensor):
        """BSCodec codec decoding.

        Args:
            codecs (torch.Tensor): neural codecs in shape (B, N_bands, T_frames).
        Returns:
            torch.Tensor: resynthesized audio.
        """
        outputs = []
        quantized = self.quantizer.decode(codes)

        for i, dec in enumerate(self.decoders):
            yi = dec(quantized[:, i, :, :]).squeeze(1)
            outputs.append(yi)

        stacked = torch.stack(outputs, dim=1)
        y_rec = reconstruct_audio_bands(stacked)
        return y_rec.unsqueeze(1)


def split_audio_bands(
    y: torch.Tensor,
    sr: int,
    bands: List[Tuple[float, float]],
    n_fft: int = 2048,
    hop_length: int = None,
    win_length: int = None,
    window_type: str = "hann",
) -> torch.Tensor:
    """
    Split a batch of audio signals into subbands via STFT masking.

    Args:
        y (Tensor): Input tensor of shape (B, T) or (T,).
        sr (int): Sampling rate.
        bands (List[Tuple[float, float]]): List of (low_freq, high_freq) in Hz.
        n_fft (int): FFT size.
        hop_length (int, optional): Hop length between frames. Defaults to n_fft // 2.
        win_length (int, optional): Window length. Defaults to n_fft.
        window_type (str): Window type, either "hann" or "hamming".

    Returns:
        Tensor: Subbands of shape (B, N_bands, T), where N_bands = len(bands).
    """
    if y.dim() == 1:
        y = y.unsqueeze(0)  # (1, T)
    B, T = y.shape

    hop_length = hop_length or (n_fft // 2)
    win_length = win_length or n_fft

    # select window
    if window_type == "hann":
        window = torch.hann_window(win_length, periodic=False, device=y.device)
    elif window_type == "hamming":
        window = torch.hamming_window(win_length, periodic=False, device=y.device)
    else:
        raise ValueError(f"Unsupported window type: {window_type}")

    # batched STFT -> (B, F, T_frames)
    S = torch.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
        center=True,
    )

    # frequency bins (F,)
    F_bins = S.size(1)
    freqs = torch.linspace(0, sr / 2, F_bins, device=y.device)

    # mask per band and ISTFT
    subbands = []
    for low_f, high_f in bands:
        mask = ((freqs >= low_f) & (freqs < high_f)).view(1, F_bins, 1)
        Sk = S * mask  # (B, F, T_frames)
        try:
            yk = torch.istft(
                Sk,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                length=T,
                center=True,
            )  # (B, T)
        except RuntimeError as err:
            if "window overlap add min" in str(err):
                try:
                    warnings.warn(
                        "ISTFT NOLA check failed for hann window; "
                        "falling back to hamming window for this band.",
                        UserWarning,
                    )
                    fallback_win1 = torch.hamming_window(
                        win_length, periodic=False, device=y.device
                    )
                    yk = torch.istft(
                        Sk,
                        n_fft=n_fft,
                        hop_length=hop_length,
                        win_length=win_length,
                        window=fallback_win1,
                        length=T,
                        center=True,
                    )
                except RuntimeError as err2:
                    warnings.warn(
                        "ISTFT NOLA check failed for hamming window; "
                        "falling back to Bartlett window for this band.",
                        UserWarning,
                    )
                    if "window overlap add min" in str(err2):
                        fallback_win2 = torch.bartlett_window(
                            win_length, periodic=False, device=y.device
                        )
                        yk = torch.istft(
                            Sk,
                            n_fft=n_fft,
                            hop_length=hop_length,
                            win_length=win_length,
                            window=fallback_win2,
                            length=T,
                            center=True,
                        )
                    else:
                        raise
            else:
                raise
        subbands.append(yk)

    # stack into (B, N_bands, T)
    return torch.stack(subbands, dim=1)


def reconstruct_audio_bands(subbands: torch.Tensor) -> torch.Tensor:
    """
    Reconstruct audio by summing subbands.

    Args:
        subbands (Tensor): Tensor of shape (B, N_bands, T).

    Returns:
        Tensor: Reconstructed audio of shape (B, T) or (T,).
    """
    y_recon = subbands.sum(dim=1)  # sum over bands -> (B, T)
    return y_recon

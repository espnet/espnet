# Copyright 2024 Yihan Wu
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""FunCodec Modules."""
import functools
import math
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from typeguard import typechecked

from espnet2.gan_codec.abs_gan_codec import AbsGANCodec
from espnet2.gan_codec.shared.decoder.seanet import SEANetDecoder  # noqa
from espnet2.gan_codec.shared.decoder.seanet_2d import SEANetDecoder2d
from espnet2.gan_codec.shared.discriminator.stft_discriminator import (
    ComplexSTFTDiscriminator,
)
from espnet2.gan_codec.shared.encoder.seanet_2d import SEANetEncoder2d
from espnet2.gan_codec.shared.loss.freq_loss import MultiScaleMelSpectrogramLoss
from espnet2.gan_codec.shared.quantizer.residual_vq import ResidualVectorQuantizer
from espnet2.gan_tts.hifigan.hifigan import (
    HiFiGANMultiPeriodDiscriminator,
    HiFiGANMultiScaleDiscriminator,
)
from espnet2.gan_tts.hifigan.loss import (
    DiscriminatorAdversarialLoss,
    FeatureMatchLoss,
    GeneratorAdversarialLoss,
)
from espnet2.torch_utils.device_funcs import force_gatherable


class FunCodec(AbsGANCodec):
    """FunCodec model."""

    @typechecked
    def __init__(
        self,
        sampling_rate: int = 24000,
        generator_params: Dict[str, Any] = {
            "hidden_dim": 128,
            "encdec_channels": 1,
            "encdec_n_filters": 32,
            "encdec_n_residual_layers": 1,
            "encdec_ratios": [(8, 1), (5, 1), (4, 1), (2, 1)],
            "encdec_activation": "ELU",
            "encdec_activation_params": {"alpha": 1.0},
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
            "decoder_trim_right_ratio": 1.0,
            "decoder_final_activation": None,
            "decoder_final_activation_params": None,
            "quantizer_n_q": 8,
            "quantizer_bins": 1024,
            "quantizer_decay": 0.99,
            "quantizer_kmeans_init": True,
            "quantizer_kmeans_iters": 50,
            "quantizer_threshold_ema_dead_code": 2,
            "quantizer_target_bandwidth": [7.5, 15],
            "quantizer_dropout": True,
            "codec_domain": ["time", "time"],
            "domain_conf": {},
        },
        discriminator_params: Dict[str, Any] = {
            "scales": 3,
            "scale_downsample_pooling": "AvgPool1d",
            "scale_downsample_pooling_params": {
                "kernel_size": 4,
                "stride": 2,
                "padding": 2,
            },
            "scale_discriminator_params": {
                "in_channels": 1,
                "out_channels": 1,
                "kernel_sizes": [15, 41, 5, 3],
                "channels": 128,
                "max_downsample_channels": 1024,
                "max_groups": 16,
                "bias": True,
                "downsample_scales": [2, 2, 4, 4, 1],
                "nonlinear_activation": "LeakyReLU",
                "nonlinear_activation_params": {"negative_slope": 0.1},
            },
            "scale_follow_official_norm": False,
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
            "complexstft_discriminator_params": {
                "in_channels": 1,
                "channels": 32,
                "strides": ((1, 2), (2, 2), (1, 2), (2, 2), (1, 2), (2, 2)),
                "chan_mults": (1, 2, 4, 4, 8, 8),
                "n_fft": 1024,
                "hop_length": 256,
                "win_length": 1024,
                "stft_normalized": False,
                "logits_abs": True,
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
        use_dual_decoder: bool = False,
        lambda_quantization: float = 1.0,
        lambda_reconstruct: float = 1.0,
        lambda_commit: float = 1.0,
        lambda_adv: float = 1.0,
        lambda_feat_match: float = 2.0,
        lambda_mel: float = 45.0,
        cache_generator_outputs: bool = False,
    ):
        """Intialize FunCodec model.

        Args:
             TODO(jiatong)
        """
        super().__init__()

        # define modules
        generator_params["encdec_ratios"] = [
            tuple(ratio) for ratio in generator_params["encdec_ratios"]
        ]
        generator_params.update(sample_rate=sampling_rate)
        self.generator = FunCodecGenerator(**generator_params)
        self.discriminator = FunCodecDiscriminator(**discriminator_params)
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
        self.lambda_quantization = lambda_quantization
        self.lambda_reconstruct = lambda_reconstruct
        self.lambda_commit = lambda_commit
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
        self.num_streams = generator_params["quantizer_n_q"]
        self.frame_shift = functools.reduce(
            lambda x, y: x * y[0], generator_params["encdec_ratios"]
        )
        self.code_size_per_stream = [
            generator_params["quantizer_bins"]
        ] * self.num_streams

    def meta_info(self) -> Dict[str, Any]:
        return {
            "fs": self.fs,
            "num_streams": self.num_streams,
            "frame_shift": self.frame_shift,
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
            return self._forward_discrminator(
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

        # TODO(jiatong): double check the multi-channel input
        audio = audio.unsqueeze(1)

        # calculate generator outputs
        reuse_cache = True
        if not self.cache_generator_outputs or self._cache is None:
            reuse_cache = False
            audio_hat, codec_commit_loss, quantization_loss, audio_hat_real = (
                self.generator(audio, use_dual_decoder=self.use_dual_decoder)
            )
        else:
            audio_hat, codec_commit_loss, quantization_loss, audio_hat_real = (
                self._cache
            )

        # store cache
        if self.training and self.cache_generator_outputs and not reuse_cache:
            self._cache = (
                audio_hat,
                codec_commit_loss,
                quantization_loss,
                audio_hat_real,
            )

        # calculate discriminator outputs
        p_hat = self.discriminator(audio_hat)
        with torch.no_grad():
            # do not store discriminator gradient in generator turn
            p = self.discriminator(audio)

        # calculate losses
        adv_loss = self.generator_adv_loss(p_hat)
        adv_loss = adv_loss * self.lambda_adv
        codec_commit_loss = codec_commit_loss * self.lambda_commit
        codec_quantization_loss = quantization_loss * self.lambda_quantization
        reconstruct_loss = (
            self.generator_reconstruct_loss(audio, audio_hat) * self.lambda_reconstruct
        )
        codec_loss = codec_commit_loss + codec_quantization_loss
        loss = adv_loss + codec_loss + reconstruct_loss
        stats = dict(
            adv_loss=adv_loss.item(),
            codec_loss=codec_loss.item(),
            codec_commit_loss=codec_commit_loss.item(),
            codec_quantization_loss=codec_quantization_loss.item(),
            reconstruct_loss=reconstruct_loss.item(),
        )
        if self.use_feat_match_loss:
            feat_match_loss = self.feat_match_loss(p_hat, p)
            feat_match_loss = feat_match_loss * self.lambda_feat_match
            loss = loss + feat_match_loss
            stats.update(feat_match_loss=feat_match_loss.item())
        if self.use_mel_loss:
            mel_loss = self.mel_loss(audio_hat, audio)
            mel_loss = self.lambda_mel * mel_loss
            loss = loss + mel_loss
            stats.update(mel_loss=mel_loss.item())
            if self.use_dual_decoder:
                mel_loss_real = self.mel_loss(audio_hat_real, audio)
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

    def _forward_discrminator(
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
            audio_hat, codec_commit_loss, codec_quantization_loss, audio_hat_real = (
                self.generator(
                    audio,
                    use_dual_decoder=self.use_dual_decoder,
                )
            )
        else:
            audio_hat, codec_commit_loss, codec_quantization_loss, audio_hat_real = (
                self._cache
            )

        # store cache
        if self.cache_generator_outputs and not reuse_cache:
            self._cache = (
                audio_hat,
                codec_commit_loss,
                codec_quantization_loss,
                audio_hat_real,
            )

        # calculate discriminator outputs
        p_hat = self.discriminator(audio_hat.detach())
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
                * wav (Tensor): Generated waveform tensor (T_wav,).
                * codec (Tensor): Generated neural codec (T_code, N_stream).

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


class FunCodecGenerator(nn.Module):
    """FunCodec generator module."""

    @typechecked
    def __init__(
        self,
        sample_rate: int = 24000,
        hidden_dim: int = 128,
        codebook_dim: int = 8,
        encdec_channels: int = 1,
        encdec_n_filters: int = 32,
        encdec_n_residual_layers: int = 1,
        encdec_ratios: List[Tuple[int, int]] = [(4, 1), (4, 1), (4, 2), (4, 1)],
        encdec_activation: str = "ELU",
        encdec_activation_params: Dict[str, Any] = {"alpha": 1.0},
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
        decoder_trim_right_ratio: float = 1.0,
        decoder_final_activation: Optional[str] = None,
        decoder_final_activation_params: Optional[dict] = None,
        quantizer_n_q: int = 8,
        quantizer_bins: int = 1024,
        quantizer_decay: float = 0.99,
        quantizer_kmeans_init: bool = True,
        quantizer_kmeans_iters: int = 50,
        quantizer_threshold_ema_dead_code: int = 2,
        quantizer_target_bandwidth: List[float] = [7.5, 15],
        quantizer_dropout: bool = True,
        codec_domain: List = ("time", "time"),
        domain_conf: Optional[Dict] = {},
        audio_normalize: bool = False,
    ):
        """Initialize FunCodec Generator.

        Args:
            TODO(jiatong)
        """
        super().__init__()

        # define domain transformation module
        self.codec_domain = codec_domain
        self.domain_conf = domain_conf
        if codec_domain[0] in ["stft", "mag_phase", "mag_angle", "mag_oracle_phase"]:
            self.enc_trans_func = torchaudio.transforms.Spectrogram(
                n_fft=domain_conf.get("n_fft", 512),
                hop_length=domain_conf.get("hop_length", 160),
                power=None,
            )
        elif codec_domain[0] in ["mag"]:
            self.enc_trans_func = torchaudio.transforms.Spectrogram(
                n_fft=domain_conf.get("n_fft", 512),
                hop_length=domain_conf.get("hop_length", 160),
                power=1,
            )
        elif codec_domain[0] == "mel":
            self.enc_trans_func = torchaudio.transforms.MelSpectrogram(
                sample_rate=target_sample_hz,  # noqa
                n_fft=domain_conf.get("n_fft", 512),
                hop_length=domain_conf.get("hop_length", 160),
                n_mels=80,
                power=2,
            )
        if codec_domain[1] in ["stft", "mag_phase", "mag_angle", "mag_oracle_phase"]:
            self.dec_trans_func = torchaudio.transforms.InverseSpectrogram(
                n_fft=domain_conf.get("n_fft", 512),
                hop_length=domain_conf.get("hop_length", 160),
            )

        # Initialize encoder
        self.encoder = SEANetEncoder2d(
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

        # Initialize quantizer
        self.quantizer = ResidualVectorQuantizer(
            dimension=hidden_dim,
            codebook_dim=codebook_dim,
            n_q=quantizer_n_q,
            bins=quantizer_bins,
            decay=quantizer_decay,
            kmeans_init=quantizer_kmeans_init,
            kmeans_iters=quantizer_kmeans_iters,
            threshold_ema_dead_code=quantizer_threshold_ema_dead_code,
            quantizer_dropout=quantizer_dropout,
        )
        self.target_bandwidths = quantizer_target_bandwidth
        self.sample_rate = sample_rate
        self.frame_rate = math.ceil(sample_rate / np.prod(encdec_ratios))

        # Initialize decoder
        self.decoder = SEANetDecoder2d(
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

        # quantization loss
        self.l1_quantization_loss = torch.nn.L1Loss(reduction="mean")
        self.l2_quantization_loss = torch.nn.MSELoss(reduction="mean")
        self.codec_domain = codec_domain
        self.domain_conf = domain_conf
        self.audio_normalize = audio_normalize

    def time_to_freq_transfer(self, x: torch.Tensor):
        if self.audio_normalize:
            mono = x.mean(dim=1, keepdim=True)
            volume = mono.pow(2).mean(dim=2, keepdim=True).sqrt()
            scale = 1e-8 + volume
            x = x / scale
            scale = scale.view(-1, 1)
        else:
            scale = None

        if self.codec_domain[0] == "stft":
            x_complex = self.enc_trans_func(x.squeeze(1))
            if self.encoder.channels == 2:
                x = torch.stack([x_complex.real, x_complex.imag], dim=1)
            else:
                x = torch.cat([x_complex.real, x_complex.imag], dim=1)
        elif self.codec_domain[0] == "mag":
            x_mag = self.enc_trans_func(x.squeeze(1))
            if self.encoder.channels == 1:
                x = x_mag.unsqueeze(1)
            else:
                x = x_mag
        elif self.codec_domain[0] == "mag_angle":
            x_complex = self.enc_trans_func(x.squeeze(1))
            x_mag = torch.abs(x_complex)
            x_log_mag = torch.log(torch.clamp(x_mag, min=1e-6))
            x_angle = torch.angle(x_complex)
            if self.encoder.channels == 2:
                x = torch.stack([x_log_mag, x_angle], dim=1)
            else:
                x = torch.cat([x_log_mag, x_angle], dim=1)
        elif self.codec_domain[0] == "mag_phase":
            x_complex = self.enc_trans_func(x.squeeze(1))
            x_mag = torch.abs(x_complex)
            x_log_mag = torch.log(torch.clamp(x_mag, min=1e-6))
            x_phase = x_complex / torch.clamp(x_mag, min=1e-6)
            if self.encoder.channels == 3:
                x = torch.stack([x_log_mag, x_phase.real, x_phase.imag], dim=1)
            else:
                x = torch.cat([x_log_mag, x_phase.real, x_phase.imag], dim=1)
        elif self.codec_domain[0] == "mel":
            x = self.enc_trans_func(x.squeeze(1))
            if self.encoder.channels == 1:
                x = x.unsqueeze(1)
        elif self.codec_domain[0] == "mag_oracle_phase":
            x_complex = self.enc_trans_func(x.squeeze(1))
            x = torch.abs(x_complex)
            if self.encoder.channels == 1:
                x = x.unsqueeze(1)
            x_phase = torch.angle(x_complex)
            scale = (scale, x_phase)
        return x, scale

    def freq_to_time_transfer(self, x: torch.Tensor, scale: torch.Tensor = None):
        if self.codec_domain[1] == "stft":
            if len(x.shape) == 3:
                out_list = torch.split(x, x.shape[1] // 2, dim=1)
            else:
                out_list = torch.split(x, 1, dim=1)
            x = torch.complex(out_list[0], out_list[1])
            x = self.dec_trans_func(x).unsqueeze(1)
        elif self.codec_domain[1] == "mag_phase":
            if len(x.shape) == 3:
                out_list = torch.split(x, x.shape[1] // 3, dim=1)
            else:
                out_list = [x.squeeze(1) for x in torch.split(x, 1, dim=1)]
            x_mag = F.softplus(out_list[0])
            x_phase = torch.complex(out_list[1], out_list[2])
            x = x_mag * x_phase
            x = self.dec_trans_func(x).unsqueeze(1)
        elif self.codec_domain[1] == "mag_angle":
            if len(x.shape) == 3:
                out_list = torch.split(x, x.shape[1] // 2, dim=1)
            else:
                out_list = [x.squeeze(1) for x in torch.split(x, 1, dim=1)]
            x_mag = F.softplus(out_list[0])
            x_angle = torch.sin(out_list[1]) * torch.pi
            x_spec = torch.complex(
                torch.cos(x_angle) * x_mag, torch.sin(x_angle) * x_mag
            )
            x = self.dec_trans_func(x_spec).unsqueeze(1)
        elif self.codec_domain[1] == "mag_oracle_phase":
            if len(x.shape) == 4:
                x = x.squeeze(1)
            (scale, x_angle), x_mag = scale, x
            x_spec = torch.complex(
                torch.cos(x_angle) * x_mag, torch.sin(x_angle) * x_mag
            )
            x = self.dec_trans_func(x_spec).unsqueeze(1)
        elif (
            self.codec_domain[0]
            in ["stft", "mag", "mag_phase", "mag_angle", "mag_oracle_phase"]
            and self.codec_domain[1] == "time"
        ):
            hop_length = self.domain_conf.get("hop_length", 160)
            x = x[:, :, hop_length // 2 : -hop_length // 2]

        if scale is not None:
            x = x * scale.view(-1, 1, 1)
        return x

    def forward(self, x: torch.Tensor, use_dual_decoder: bool = False):
        """FunCodec forward propagation.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 1, T).
            use_dual_decoder (bool): Whether to use dual decoder for encoder out
        Returns:
            torch.Tensor: resynthesized audio.
            torch.Tensor: commitment loss.
            torch.Tensor: quantization loss
            torch.Tensor: resynthesized audio from encoder.
        """
        x, scale = self.time_to_freq_transfer(x)
        encoder_out = self.encoder(x)
        max_idx = len(self.target_bandwidths) - 1

        # randomly pick up one bandwidth
        bw = self.target_bandwidths[random.randint(0, max_idx)]

        # Forward quantizer
        quantized, _, _, commit_loss, quantization_loss = self.quantizer(
            encoder_out, self.frame_rate, bw
        )
        # quantization_loss = self.l1_quantization_loss(
        #     encoder_out, quantized.detach()
        # ) + self.l2_quantization_loss(encoder_out, quantized.detach())

        resyn_audio = self.decoder(quantized)[:, :, :, : x.shape[3]]
        resyn_audio = self.freq_to_time_transfer(resyn_audio)
        if use_dual_decoder:
            resyn_audio_real = self.decoder(encoder_out)
        else:
            resyn_audio_real = None
        return resyn_audio, commit_loss, quantization_loss, resyn_audio_real

    def encode(
        self,
        x: torch.Tensor,
        target_bw: Optional[float] = None,
    ):
        """FunCodec codec encoding.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 1, T).
        Returns:
            torch.Tensor: neural codecs in shape ().
        """

        x, scale = self.time_to_freq_transfer(x)
        encoder_out = self.encoder(x)
        if target_bw is None:
            bw = self.target_bandwidths[-1]
        else:
            bw = target_bw
        codes = self.quantizer.encode(encoder_out, self.frame_rate, bw)
        return codes

    def decode(self, codes: torch.Tensor):
        """FunCodec codec decoding.

        Args:
            codecs (torch.Tensor): neural codecs in shape ().
        Returns:
            torch.Tensor: resynthesized audio.
        """
        quantized = self.quantizer.decode(codes)
        resyn_audio = self.decoder(quantized)
        resyn_audio = self.freq_to_time_transfer(resyn_audio)
        return resyn_audio


class FunCodecDiscriminator(nn.Module):
    """FunCodec discriminator module."""

    def __init__(
        self,
        # Multi-scale discriminator related
        scales: int = 3,
        scale_downsample_pooling: str = "AvgPool1d",
        # follow the official implementation setting
        scale_downsample_pooling_params: Dict[str, Any] = {
            "kernel_size": 4,
            "stride": 2,
            "padding": 2,
        },
        scale_discriminator_params: Dict[str, Any] = {
            "in_channels": 1,
            "out_channels": 1,
            "kernel_sizes": [15, 41, 5, 3],
            "channels": 128,
            "max_downsample_channels": 1024,
            "max_groups": 16,
            "bias": True,
            "downsample_scales": [2, 2, 4, 4, 1],
            "nonlinear_activation": "LeakyReLU",
            "nonlinear_activation_params": {"negative_slope": 0.1},
        },
        scale_follow_official_norm: bool = False,
        # Multi period discriminator related
        periods: List[int] = [2, 3, 5, 7, 11],
        period_discriminator_params: Dict[str, Any] = {
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
        # ComplexSTFT discriminator related
        complexstft_discriminator_params: Dict[str, Any] = {
            "in_channels": 1,
            "channels": 32,
            "strides": [[1, 2], [2, 2], [1, 2], [2, 2], [1, 2], [2, 2]],
            "chan_mults": [1, 2, 4, 4, 8, 8],
            "n_fft": 1024,
            "hop_length": 256,
            "win_length": 1024,
            "stft_normalized": False,
        },
    ):
        """Initialize FunCodec Discriminator module.

        Args:
            scales (int): Number of multi-scales.
            sclae_downsample_pooling (str): Pooling module name for downsampling of the
                inputs.
            scale_downsample_pooling_params (Dict[str, Any]): Parameters for the above
                pooling module.
            scale_discriminator_params (Dict[str, Any]): Parameters for hifi-gan  scale
                discriminator module.
            scale_follow_official_norm (bool): Whether to follow the norm setting of the
                official implementaion. The first discriminator uses spectral norm
                and the other discriminators use weight norm.
            complexstft_discriminator_params (Dict[str, Any]): Parameters for the
                complex stft discriminator module.
        """
        super().__init__()

        self.msd = HiFiGANMultiScaleDiscriminator(
            scales=scales,
            downsample_pooling=scale_downsample_pooling,
            downsample_pooling_params=scale_downsample_pooling_params,
            discriminator_params=scale_discriminator_params,
            follow_official_norm=scale_follow_official_norm,
        )
        self.mpd = HiFiGANMultiPeriodDiscriminator(
            periods=periods,
            discriminator_params=period_discriminator_params,
        )
        self.complex_stft_d = ComplexSTFTDiscriminator(
            **complexstft_discriminator_params
        )

    def forward(self, x: torch.Tensor) -> List[List[torch.Tensor]]:
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).

        Returns:
            List[List[Tensor]]: List of list of each discriminator outputs,
                which consists of each layer output tensors. Multi scale and
                multi period ones are concatenated.

        """
        msd_outs = self.msd(x)
        mpd_outs = self.mpd(x)
        complex_stft_outs = self.complex_stft_d(x)
        return msd_outs + mpd_outs + complex_stft_outs

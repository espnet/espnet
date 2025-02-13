# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""SpeechTokenizer Modules."""
import copy
import functools
import logging
import math
import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typeguard import typechecked

from espnet2.gan_codec.abs_gan_codec import AbsGANCodec
from espnet2.gan_codec.shared.decoder.seanet import SEANetDecoder
from espnet2.gan_codec.shared.discriminator.msmpmb_discriminator import (
    MultiScaleMultiPeriodMultiBandDiscriminator,
)
from espnet2.gan_codec.shared.discriminator.msstft_discriminator import (
    MultiScaleSTFTDiscriminator,
)
from espnet2.gan_codec.shared.encoder.seanet import SEANetEncoder
from espnet2.gan_codec.shared.loss.freq_loss import MultiScaleMelSpectrogramLoss
from espnet2.gan_codec.shared.loss.loss_balancer import Balancer
from espnet2.gan_codec.shared.quantizer.residual_vq import ResidualVectorQuantizer
from espnet2.gan_tts.hifigan.loss import (
    DiscriminatorAdversarialLoss,
    FeatureMatchLoss,
    GeneratorAdversarialLoss,
)
from espnet2.torch_utils.device_funcs import force_gatherable


class SpeechTokenizer(AbsGANCodec):
    """ "SpeechTokenizer model."""

    @typechecked
    def __init__(
        self,
        sampling_rate: int = 24000,
        generator_params: Dict[str, Any] = {
            "hidden_dim": 128,
            "semantic_dim": 1024,
            "semantic_type": "espnet",
            "semantic_model": "hubert",
            "semantic_sample_rate": 16000,
            "semantic_layer": 17,
            "encdec_channels": 1,
            "encdec_n_filters": 32,
            "encdec_n_residual_layers": 1,
            "encdec_ratios": [8, 5, 4, 2],
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
        },
        discriminator_params: Dict[str, Any] = {
            "msstft_discriminator_params": {
                "filters": 32,
                "in_channels": 1,
                "out_channels": 1,
                "sep_channels": False,
                "norm": "weight_norm",
                "n_ffts": [1024, 2048, 512, 256, 128],
                "hop_lengths": [256, 512, 128, 64, 32],
                "win_lengths": [1024, 2048, 512, 256, 128],
                "activation": "LeakyReLU",
                "activation_params": {"negative_slope": 0.3},
            },
            "msmpmb_discriminator_params": {
                "rates": [],
                "periods": [2, 3, 5, 7, 11],
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
        lambda_commit: float = 1.0,
        lambda_adv: float = 1.0,
        lambda_feat_match: float = 2.0,
        lambda_mel: float = 45.0,
        lambda_semantic: float = 1.0,
        cache_generator_outputs: bool = False,
        use_loss_balancer: bool = False,
        balance_ema_decay: float = 0.99,
    ):
        """Intialize SpeechTokenizer model.

        Args:
             TODO(jiatong)
        """
        super().__init__()

        # define modules
        generator_params.update(sample_rate=sampling_rate)
        self.generator = SpeechTokenizerGenerator(**generator_params)
        self.discriminator = SpeechTokenizerDiscriminator(**discriminator_params)
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
        self.lambda_semantic = lambda_semantic
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
            lambda x, y: x * y, generator_params["encdec_ratios"]
        )
        self.code_size_per_stream = [
            generator_params["quantizer_bins"]
        ] * self.num_streams

        # loss balancer
        if use_loss_balancer:
            self.loss_balancer = Balancer(
                ema_decay=balance_ema_decay,
                per_batch_item=True,
            )
        else:
            self.loss_balancer = None

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
            (
                audio_hat,
                codec_commit_loss,
                quantization_loss,
                semantic_loss,
                audio_hat_real,
            ) = self.generator(audio, use_dual_decoder=self.use_dual_decoder)
        else:
            (
                audio_hat,
                codec_commit_loss,
                quantization_loss,
                semantic_loss,
                audio_hat_real,
            ) = self._cache

        # store cache
        if self.training and self.cache_generator_outputs and not reuse_cache:
            self._cache = (
                audio_hat,
                codec_commit_loss,
                quantization_loss,
                semantic_loss,
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
        semantic_loss = semantic_loss * self.lambda_semantic
        codec_loss = codec_commit_loss + codec_quantization_loss
        loss = adv_loss + codec_loss + reconstruct_loss
        stats = dict(
            adv_loss=adv_loss.item(),
            codec_loss=codec_loss.item(),
            codec_commit_loss=codec_commit_loss.item(),
            codec_quantization_loss=codec_quantization_loss.item(),
            reconstruct_loss=reconstruct_loss.item(),
            semantic_loss=semantic_loss.item(),
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

        if self.loss_balancer is not None and self.training:
            # any loss built on audio_hat is processed by balancer
            balanced_losses = {
                "reconstruct": reconstruct_loss,
                "adv": adv_loss,
            }
            if self.use_feat_match_loss:
                balanced_losses.update(feat_match=feat_match_loss)
            if self.use_mel_loss:
                balanced_losses.update(mel=mel_loss)

            balanced_loss, norm_stats = self.loss_balancer(balanced_losses, audio_hat)
            stats.update(norm_stats)

            loss = sum(balanced_loss.values()) + codec_loss
            if self.use_mel_loss and self.use_dual_decoder:
                loss = loss + mel_loss_real

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


class SpeechTokenizerGenerator(nn.Module):
    """SpeechTokenizer generator module."""

    @typechecked
    def __init__(
        self,
        sample_rate: int = 24000,
        hidden_dim: int = 128,
        semantic_dim: int = 1024,
        semantic_type: str = "espnet",
        semantic_model: str = "hubert",
        semantic_sample_rate: int = 16000,
        semantic_layer: int = 17,
        semantic_loss: str = "cosine",
        encdec_channels: int = 1,
        encdec_n_filters: int = 32,
        encdec_n_residual_layers: int = 1,
        encdec_ratios: List[int] = [8, 5, 4, 2],
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
        encdec_bidirectional_lstm: bool = False,
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
    ):
        """Initialize SpeechTokenizer Generator.

        Args:
            TODO(jiatong)
        """
        super().__init__()

        # Initialize encoder
        self.encoder = SEANetEncoder(
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
            bidirectional_lstm=encdec_bidirectional_lstm,
        )

        # Initialize quantizer
        self.quantizer = ResidualVectorQuantizer(
            dimension=hidden_dim,
            n_q=quantizer_n_q,
            bins=quantizer_bins,
            decay=quantizer_decay,
            kmeans_init=quantizer_kmeans_init,
            kmeans_iters=quantizer_kmeans_iters,
            threshold_ema_dead_code=quantizer_threshold_ema_dead_code,
        )
        self.target_bandwidths = quantizer_target_bandwidth
        self.sample_rate = sample_rate
        self.frame_rate = math.ceil(sample_rate / np.prod(encdec_ratios))

        # Initialize decoder
        self.decoder = SEANetDecoder(
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

        # Semantic related
        self.semantic_prediction = nn.Linear(hidden_dim, semantic_dim)
        self.semantic_type = semantic_type
        self.semantic_sample_rate = semantic_sample_rate
        self.semantic_layer = semantic_layer
        self.semantic_loss = semantic_loss
        assert semantic_type in ["espnet", "s3prl"]
        if semantic_type == "espnet":
            from espnet2.tasks.hubert import HubertTask

            self.semantic, _ = HubertTask.build_model_from_file(
                None, semantic_model, device=self.encoder.device
            )
        elif semantic_type == "s3prl":
            from s3prl.nn import S3PRLUpstream

            self.semantic = S3PRLUpstream(semantic_model).to(self.encoder.device)
        self.semantic.eval()

        # quantization loss
        self.l1_quantization_loss = torch.nn.L1Loss(reduction="mean")
        self.l2_quantization_loss = torch.nn.MSELoss(reduction="mean")

    def forward(self, x: torch.Tensor, use_dual_decoder: bool = False):
        """SpeechTokenizer forward propagation.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 1, T).
            use_dual_decoder (bool): Whether to use dual decoder for encoder out
        Returns:
            torch.Tensor: resynthesized audio.
            torch.Tensor: commitment loss.
            torch.Tensor: quantization loss
            torch.Tensor: resynthesized audio from encoder.
        """
        with torch.no_grad():
            if self.sample_rate != self.semantic_sample_rate:
                if self.sample_rate < self.semantic_sample_rate:
                    raise ValueError(
                        "Semantic model sample rate is more than encoder sample rate, likely a bug"
                    )
                semantic_audio = F.resample(
                    x,
                    self.sample_rate,
                    self.semantic_sample_rate,
                    resampling_method="sinc_interp_hann",
                )
            else:
                semantic_audio = x
            semantic_max_len = semantic_audio.size(2)
            semantic_seq_len = torch.tensor(
                [semantic_max_len] * semantic_audio.size(0)
            ).to(semantic_audio.device)
            if self.semantic_type == "espnet":
                semantic = self.semantic(semantic_audio.squeeze(1), semantic_seq_len)
                if self.semantic_layer == -1:
                    semantic = semantic.mean(dim=1)
                else:
                    assert self.semantic_layer < semantic.size(
                        1
                    ), "semantic layer out of range"
                    semantic = semantic[:, self.semantic_layer]
            elif self.semantic_type == "s3prl":
                semantic = self.semantic(semantic_audio.squeeze(1), semantic_seq_len)
                if self.semantic_layer == -1:
                    semantic = torch.stack(semantic).mean(dim=0)
                else:
                    assert self.semantic_layer < semantic.size(
                        1
                    ), "semantic layer out of range"
                    semantic = semantic[self.semantic_layer]

        encoder_out = self.encoder(x)
        max_idx = len(self.target_bandwidths) - 1

        # randomly pick up one bandwidth
        bw = self.target_bandwidths[random.randint(0, max_idx)]

        # Forward quantizer
        quantized_list, _, _, commit_loss = self.quantizer(
            encoder_out, self.frame_rate, bw, return_list=True
        )

        semantic_stream = quantized_list[0]
        quantized = quantized_list[-1]

        quantization_loss = self.l1_quantization_loss(
            encoder_out, quantized.detach()
        ) + self.l2_quantization_loss(encoder_out, quantized.detach())

        # calculate semantic feature
        semantic_prediction = self.semantic_prediction(semantic_stream)
        min_len = semantic_prediction.size(1), semantic.size(1)
        if self.semantic_loss == "L1":
            semantic_loss = F.l1_loss(
                semantic_prediction[:, :min_len],
                semantic[:, :min_len],
                reduction="mean",
            )
        elif self.semantic_loss == "L2":
            semantic_loss = F.mse_loss(
                semantic_prediction[:, :min_len],
                semantic[:, :min_len],
                reduction="mean",
            )
        elif self.semantic_loss == "cosine":
            semantic_loss = -torch.log(
                0.5
                + 1e-6
                - F.cosine_similarity(
                    semantic_prediction[:, :min_len], semantic[:, :min_len], axis=1
                )
                / 2
            ).mean()
        else:
            raise ValueError(
                "Unsupported semantic loss type [{}]. Supported types are L1, L2, cosine".format(
                    self.semantic_loss
                )
            )

        resyn_audio = self.decoder(quantized)

        if use_dual_decoder:
            resyn_audio_real = self.decoder(encoder_out)
        else:
            resyn_audio_real = None
        return (
            resyn_audio,
            commit_loss,
            quantization_loss,
            semantic_loss,
            resyn_audio_real,
        )

    def encode(
        self,
        x: torch.Tensor,
        target_bw: Optional[float] = None,
    ):
        """SpeechTokenizer codec encoding.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 1, T).
        Returns:
            torch.Tensor: neural codecs in shape ().
        """

        encoder_out = self.encoder(x)
        if target_bw is None:
            bw = self.target_bandwidths[-1]
        else:
            bw = target_bw
        codes = self.quantizer.encode(encoder_out, self.frame_rate, bw)
        return codes

    def decode(self, codes: torch.Tensor):
        """SpeechTokenizer codec decoding.

        Args:
            codecs (torch.Tensor): neural codecs in shape ().
        Returns:
            torch.Tensor: resynthesized audio.
        """
        quantized = self.quantizer.decode(codes)
        resyn_audio = self.decoder(quantized)
        return resyn_audio


class SpeechTokenizerDiscriminator(nn.Module):
    """SpeechTokenizer discriminator module."""

    def __init__(
        self,
        msstft_discriminator_params={
            "filters": 32,
            "in_channels": 1,
            "out_channels": 1,
            "sep_channels": False,
            "norm": "weight_norm",
            "n_ffts": [1024, 2048, 512, 256, 128],
            "hop_lengths": [256, 512, 128, 64, 32],
            "win_lengths": [1024, 2048, 512, 256, 128],
            "activation": "LeakyReLU",
            "activation_params": {"negative_slope": 0.3},
        },
        msmpmb_discriminator_params={
            "rates": [],
            "periods": [2, 3, 5, 7, 11],
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
    ):
        """Initialize SpeechTokenizer Discriminator module.

        Args:
            TODO(jiatong)
        """
        super().__init__()

        self.msmpmb_discriminator = MultiScaleMultiPeriodMultiBandDiscriminator(
            **msmpmb_discriminator_params
        )
        self.msstft_discriminator = MultiScaleSTFTDiscriminator(
            **msstft_discriminator_params
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
        msmpmb_outs = self.msmpmb_discriminator(x)
        msstft_outs = self.msstft_discriminator(x)
        return msmpmb_outs + msstft_outs

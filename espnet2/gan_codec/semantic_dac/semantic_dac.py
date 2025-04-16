# Copyright 2024 Yihan Wu
# Copyright 2025 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""DAC Modules with semantic features."""
import copy
import functools
import logging
import math
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from typeguard import typechecked

from espnet2.gan_codec.abs_gan_codec import AbsGANCodec
from espnet2.gan_codec.shared.decoder.seanet import SEANetDecoder
from espnet2.gan_codec.dac.dac import DACDiscriminator
from espnet2.gan_codec.shared.encoder.seanet import SEANetEncoder
from espnet2.gan_codec.shared.loss.freq_loss import MultiScaleMelSpectrogramLoss
from espnet2.gan_codec.shared.quantizer.residual_vq import ResidualVectorQuantizer
from espnet2.gan_tts.hifigan.loss import (
    DiscriminatorAdversarialLoss,
    FeatureMatchLoss,
    GeneratorAdversarialLoss,
)
from espnet2.torch_utils.device_funcs import force_gatherable

logger = logging.getLogger(__name__)


class SemanticDAC(AbsGANCodec):
    """DAC model with semantic features.

    This model uses a semantic encoder to extract features from audio,
    which are then used in the training process to improve codec quality.
    """

    @typechecked
    def __init__(
        self,
        sampling_rate: int = 24000,
        generator_params: Dict[str, Any] = {
            "hidden_dim": 128,
            "semantic_dim": 768,
            "semantic_type": "s3prl",
            "semantic_model": "hubert",
            "semantic_sample_rate": 16000,
            "semantic_layer": 9,
            "encdec_channels": 1,
            "encdec_n_filters": 32,
            "encdec_n_residual_layers": 1,
            "encdec_ratios": [8, 6, 5, 2],
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
        },
        discriminator_params: Dict[str, Any] = {
            "scale_follow_official_norm": False,
            "msmpmb_discriminator_params": {
                "rates": [],
                "periods": [2, 3, 5, 7, 11],
                "fft_sizes": [2048, 1024, 512],
                "sample_rate": 24000,
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
    ):
        """Initialize DAC model.

        Args:
            sampling_rate: The sample rate of the input audio.
            generator_params: Parameters for the generator model.
            discriminator_params: Parameters for the discriminator model.
            generator_adv_loss_params: Parameters for generator adversarial loss.
            discriminator_adv_loss_params:
                Parameters for discriminator adversarial loss.
            use_feat_match_loss: Whether to use feature matching loss.
            feat_match_loss_params: Parameters for feature matching loss.
            use_mel_loss: Whether to use mel-spectrogram loss.
            mel_loss_params: Parameters for mel-spectrogram loss.
            use_dual_decoder: Whether to use dual decoder mode.
            lambda_quantization: Weight for quantization loss.
            lambda_reconstruct: Weight for reconstruction loss.
            lambda_commit: Weight for commitment loss.
            lambda_adv: Weight for adversarial loss.
            lambda_feat_match: Weight for feature matching loss.
            lambda_mel: Weight for mel-spectrogram loss.
            lambda_semantic: Weight for semantic loss.
            cache_generator_outputs: Whether to cache generator outputs.
        """
        super().__init__()

        # Update sample rate for all components
        generator_params["sample_rate"] = sampling_rate

        if "sample_rate" in discriminator_params.get("msmpmb_discriminator_params", {}):
            discriminator_params["msmpmb_discriminator_params"][
                "sample_rate"
            ] = sampling_rate

        if use_mel_loss:
            mel_loss_params["fs"] = sampling_rate

        # Define modules
        self.generator = SemanticDACGenerator(**generator_params)
        self.discriminator = DACDiscriminator(**discriminator_params)

        # Define loss functions
        self.generator_adv_loss = GeneratorAdversarialLoss(**generator_adv_loss_params)
        self.generator_reconstruct_loss = nn.L1Loss(reduction="mean")
        self.discriminator_adv_loss = DiscriminatorAdversarialLoss(
            **discriminator_adv_loss_params
        )

        # Optional losses
        self.use_feat_match_loss = use_feat_match_loss
        if self.use_feat_match_loss:
            self.feat_match_loss = FeatureMatchLoss(**feat_match_loss_params)

        self.use_mel_loss = use_mel_loss
        if self.use_mel_loss:
            self.mel_loss = MultiScaleMelSpectrogramLoss(**mel_loss_params)

        # Handle dual decoder mode
        self.use_dual_decoder = use_dual_decoder
        if self.use_dual_decoder and not self.use_mel_loss:
            logger.warning(
                "Dual decoder is enabled but Mel loss is disabled."
                "This configuration is ineffective."
            )
            self.use_dual_decoder = False

        # Loss coefficients
        self.lambda_quantization = lambda_quantization
        self.lambda_reconstruct = lambda_reconstruct
        self.lambda_commit = lambda_commit
        self.lambda_semantic = lambda_semantic
        self.lambda_adv = lambda_adv
        self.lambda_feat_match = lambda_feat_match if use_feat_match_loss else 0.0
        self.lambda_mel = lambda_mel if use_mel_loss else 0.0

        # Cache settings
        self.cache_generator_outputs = cache_generator_outputs
        self._cache = None

        # Meta information
        self.fs = sampling_rate
        self.num_streams = generator_params["quantizer_n_q"]
        self.frame_shift = functools.reduce(
            lambda x, y: x * y, generator_params["encdec_ratios"]
        )
        self.code_size_per_stream = [
            generator_params["quantizer_bins"]
        ] * self.num_streams

    def meta_info(self) -> Dict[str, Any]:
        """Return model meta information.

        Returns:
            Dict with model information.
        """
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
        """Perform forward pass.

        Args:
            audio: Audio waveform tensor (B, T_wav).
            forward_generator: Whether to forward generator.

        Returns:
            Dict with loss, stats, weight, and optimizer index.
        """
        if forward_generator:
            return self._forward_generator(audio=audio, **kwargs)
        else:
            return self._forward_discriminator(audio=audio, **kwargs)

    def _forward_generator(
        self,
        audio: torch.Tensor,
        **kwargs,
    ) -> Dict[str, Any]:
        """Perform generator forward pass.

        Args:
            audio: Audio waveform tensor (B, T_wav).

        Returns:
            Dict with loss, stats, weight, and optimizer index.
        """
        # Setup
        batch_size = audio.size(0)

        # Add channel dimension if needed
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)

        # Calculate generator outputs
        reuse_cache = True
        if not self.cache_generator_outputs or self._cache is None:
            reuse_cache = False
            outputs = self.generator(audio, use_dual_decoder=self.use_dual_decoder)
            (
                audio_hat,
                codec_commit_loss,
                quantization_loss,
                semantic_loss,
                audio_hat_real,
            ) = outputs
        else:
            (
                audio_hat,
                codec_commit_loss,
                quantization_loss,
                semantic_loss,
                audio_hat_real,
            ) = self._cache

        # Store cache
        if self.training and self.cache_generator_outputs and not reuse_cache:
            self._cache = (
                audio_hat,
                codec_commit_loss,
                quantization_loss,
                semantic_loss,
                audio_hat_real,
            )

        # Calculate discriminator outputs
        p_hat = self.discriminator(audio_hat)
        with torch.no_grad():
            # Do not store discriminator gradient in generator turn
            p = self.discriminator(audio)

        # Calculate losses
        adv_loss = self.generator_adv_loss(p_hat) * self.lambda_adv
        codec_commit_loss = codec_commit_loss * self.lambda_commit
        codec_quantization_loss = quantization_loss * self.lambda_quantization
        reconstruct_loss = (
            self.generator_reconstruct_loss(audio, audio_hat) * self.lambda_reconstruct
        )
        semantic_loss = semantic_loss * self.lambda_semantic

        codec_loss = codec_commit_loss + codec_quantization_loss
        loss = adv_loss + codec_loss + reconstruct_loss + semantic_loss

        # Collect statistics
        stats = {
            "adv_loss": adv_loss.item(),
            "codec_loss": codec_loss.item(),
            "semantic_loss": semantic_loss.item(),
            "codec_commit_loss": codec_commit_loss.item(),
            "codec_quantization_loss": codec_quantization_loss.item(),
            "reconstruct_loss": reconstruct_loss.item(),
        }

        # Add feature matching loss if enabled
        if self.use_feat_match_loss:
            feat_match_loss = self.feat_match_loss(p_hat, p) * self.lambda_feat_match
            loss = loss + feat_match_loss
            stats["feat_match_loss"] = feat_match_loss.item()

        # Add mel-spectrogram loss if enabled
        if self.use_mel_loss:
            mel_loss = self.mel_loss(audio_hat, audio) * self.lambda_mel
            loss = loss + mel_loss
            stats["mel_loss"] = mel_loss.item()

            if self.use_dual_decoder and audio_hat_real is not None:
                mel_loss_real = self.mel_loss(audio_hat_real, audio) * self.lambda_mel
                loss = loss + mel_loss_real
                stats["mel_loss_real"] = mel_loss_real.item()

        stats["loss"] = loss.item()

        # Make values gatherable across devices
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)

        # Reset cache if needed
        if reuse_cache or not self.training:
            self._cache = None

        return {
            "loss": loss,
            "stats": stats,
            "weight": weight,
            "optim_idx": 0,  # Needed for trainer
        }

    def _forward_discriminator(
        self,
        audio: torch.Tensor,
        **kwargs,
    ) -> Dict[str, Any]:
        """Perform discriminator forward pass.

        Args:
            audio: Audio waveform tensor (B, T_wav).

        Returns:
            Dict with loss, stats, weight, and optimizer index.
        """
        # Setup
        batch_size = audio.size(0)

        # Add channel dimension if needed
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)

        # Calculate generator outputs
        reuse_cache = True
        if not self.cache_generator_outputs or self._cache is None:
            reuse_cache = False
            outputs = self.generator(audio, use_dual_decoder=self.use_dual_decoder)
            (
                audio_hat,
                codec_commit_loss,
                codec_quantization_loss,
                semantic_loss,
                audio_hat_real,
            ) = outputs
        else:
            (
                audio_hat,
                codec_commit_loss,
                codec_quantization_loss,
                semantic_loss,
                audio_hat_real,
            ) = self._cache

        # Store cache
        if self.cache_generator_outputs and not reuse_cache:
            self._cache = (
                audio_hat,
                codec_commit_loss,
                codec_quantization_loss,
                semantic_loss,
                audio_hat_real,
            )

        # Calculate discriminator outputs
        p_hat = self.discriminator(
            audio_hat.detach()
        )  # Detach to avoid grad flow to generator
        p = self.discriminator(audio)

        # Calculate losses
        real_loss, fake_loss = self.discriminator_adv_loss(p_hat, p)
        loss = real_loss + fake_loss

        # Collect statistics
        stats = {
            "discriminator_loss": loss.item(),
            "real_loss": real_loss.item(),
            "fake_loss": fake_loss.item(),
        }

        # Make values gatherable across devices
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)

        # Reset cache if needed
        if reuse_cache or not self.training:
            self._cache = None

        return {
            "loss": loss,
            "stats": stats,
            "weight": weight,
            "optim_idx": 1,  # Needed for trainer
        }

    def inference(
        self,
        x: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Run inference.

        Args:
            x: Input audio (T_wav,).

        Returns:
            Dict with generated waveform and neural codec.
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
            x: Input audio (T_wav,).

        Returns:
            Generated codes (T_code, N_stream).
        """
        return self.generator.encode(x)

    def decode(
        self,
        x: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Run decoding.

        Args:
            x: Input codes (T_code, N_stream).

        Returns:
            Generated waveform (T_wav,).
        """
        return self.generator.decode(x)


class SemanticDACGenerator(nn.Module):
    """DAC generator module with semantic features."""

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
        codebook_dim: int = 8,
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
    ):
        """Initialize DAC Generator.

        Args:
            sample_rate: Sample rate of input audio.
            hidden_dim: Hidden dimension for models.
            semantic_dim: Dimension of semantic features.
            semantic_type: Type of semantic model ('espnet' or 's3prl').
            semantic_model: Name of semantic model.
            semantic_sample_rate: Sample rate for semantic model.
            semantic_layer: Which layer to extract from semantic model.
            semantic_loss: Type of semantic loss ('L1', 'L2', or 'cosine').
            codebook_dim: Dimension of codebook.
            encdec_channels: Number of encoder/decoder channels.
            encdec_n_filters: Number of filters in encoder/decoder.
            encdec_n_residual_layers: Number of residual layers.
            encdec_ratios: Upsampling/downsampling ratios.
            encdec_activation: Activation function.
            encdec_activation_params: Parameters for activation function.
            encdec_norm: Normalization method.
            encdec_norm_params: Parameters for normalization method.
            encdec_kernel_size: Kernel size.
            encdec_residual_kernel_size: Residual kernel size.
            encdec_last_kernel_size: Last kernel size.
            encdec_dilation_base: Base for dilation calculation.
            encdec_causal: Whether to use causal convolution.
            encdec_pad_mode: Padding mode.
            encdec_true_skip: Whether to use true skip connections.
            encdec_compress: Compression factor.
            encdec_lstm: Number of LSTM layers.
            decoder_trim_right_ratio: Trim ratio for decoder output.
            decoder_final_activation: Final activation function.
            decoder_final_activation_params: Parameters for final activation.
            quantizer_n_q: Number of quantizers.
            quantizer_bins: Number of bins per quantizer.
            quantizer_decay: Decay factor for EMA updates.
            quantizer_kmeans_init: Whether to initialize with k-means.
            quantizer_kmeans_iters: Number of k-means iterations.
            quantizer_threshold_ema_dead_code: Threshold for resetting dead codes.
            quantizer_target_bandwidth: Target bandwidth ranges.
            quantizer_dropout: Whether to use dropout in quantizer.
        """
        super().__init__()

        # Validate parameters
        self._validate_parameters(
            semantic_type=semantic_type,
            semantic_loss=semantic_loss,
            sample_rate=sample_rate,
            semantic_sample_rate=semantic_sample_rate,
        )

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

        # Set model parameters
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

        # Semantic related components
        self.semantic_prediction = nn.Linear(hidden_dim, semantic_dim)
        self.semantic_type = semantic_type
        self.semantic_sample_rate = semantic_sample_rate
        self.semantic_layer = semantic_layer
        self.semantic_loss = semantic_loss

        # Initialize semantic model
        self._initialize_semantic_model(semantic_type, semantic_model)

        # Quantization loss functions
        self.l1_quantization_loss = nn.L1Loss(reduction="mean")
        self.l2_quantization_loss = nn.MSELoss(reduction="mean")

    def _validate_parameters(
        self,
        semantic_type: str,
        semantic_loss: str,
        sample_rate: int,
        semantic_sample_rate: int,
    ) -> None:
        """Validate input parameters.

        Args:
            semantic_type: Type of semantic model.
            semantic_loss: Type of semantic loss.
            sample_rate: Sample rate of input audio.
            semantic_sample_rate: Sample rate for semantic model.

        Raises:
            ValueError: If parameters are invalid.
        """
        if semantic_type not in ["espnet", "s3prl"]:
            raise ValueError(
                f"Unsupported semantic type: {semantic_type}. Use 'espnet' or 's3prl'."
            )

        if semantic_loss not in ["L1", "L2", "cosine"]:
            raise ValueError(
                f"Unsupported semantic loss: {semantic_loss}."
                " Use 'L1', 'L2', or 'cosine'."
            )

        if sample_rate < semantic_sample_rate:
            raise ValueError(
                "Semantic model sample rate is higher than encoder sample rate. "
                f"Got sample_rate={sample_rate},"
                f" semantic_sample_rate={semantic_sample_rate}."
            )

    def _initialize_semantic_model(
        self, semantic_type: str, semantic_model: str
    ) -> None:
        """Initialize semantic model.

        Args:
            semantic_type: Type of semantic model.
            semantic_model: Name of semantic model.
        """
        if semantic_type == "espnet":
            from espnet2.tasks.hubert import HubertTask

            self.semantic, _ = HubertTask.build_model_from_file(
                None, semantic_model, device="cpu"
            )
        elif semantic_type == "s3prl":
            from s3prl.nn import S3PRLUpstream

            self.semantic = S3PRLUpstream(semantic_model)

        # Set to evaluation mode
        self.semantic.eval()

        # Freeze semantic model parameters
        for param in self.semantic.parameters():
            param.requires_grad = False

    def _extract_semantic_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract semantic features from audio.

        Args:
            x: Input audio tensor.

        Returns:
            Semantic features.
        """
        with torch.no_grad():
            # Resample if needed
            if self.sample_rate != self.semantic_sample_rate:
                semantic_audio = torchaudio.functional.resample(
                    x,
                    self.sample_rate,
                    self.semantic_sample_rate,
                    resampling_method="sinc_interp_hann",
                )
            else:
                semantic_audio = x

            # Prepare for semantic model
            semantic_max_len = semantic_audio.size(2)
            semantic_seq_len = torch.tensor(
                [semantic_max_len] * semantic_audio.size(0)
            ).to(semantic_audio.device)

            # Process through semantic model
            if self.semantic_type == "espnet":
                semantic = self.semantic(semantic_audio.squeeze(1), semantic_seq_len)
                if self.semantic_layer == -1:
                    semantic = semantic.mean(dim=1)
                else:
                    assert self.semantic_layer < semantic.size(1), (
                        f"Semantic layer {self.semantic_layer} out of range "
                        f"for model with {semantic.size(1)} layers"
                    )
                    semantic = semantic[:, self.semantic_layer]
            elif self.semantic_type == "s3prl":
                semantic, _ = self.semantic(semantic_audio.squeeze(1), semantic_seq_len)
                if self.semantic_layer == -1:
                    semantic = torch.stack(semantic).mean(dim=0)
                else:
                    assert self.semantic_layer < len(semantic), (
                        f"Semantic layer {self.semantic_layer} out of range "
                        f"for model with {len(semantic)} layers"
                    )
                    semantic = semantic[self.semantic_layer]

        return semantic

    def _calculate_semantic_loss(
        self, semantic_prediction: torch.Tensor, semantic: torch.Tensor
    ) -> torch.Tensor:
        """Calculate semantic loss.

        Args:
            semantic_prediction: Predicted semantic features.
            semantic: Target semantic features.

        Returns:
            Semantic loss.
        """
        # Find minimum length between prediction and target
        min_len = min(semantic_prediction.size(1), semantic.size(1))

        # Compute loss based on the specified type
        if self.semantic_loss == "L1":
            loss = F.l1_loss(
                semantic_prediction[:, :min_len],
                semantic[:, :min_len],
                reduction="mean",
            )
        elif self.semantic_loss == "L2":
            loss = F.mse_loss(
                semantic_prediction[:, :min_len],
                semantic[:, :min_len],
                reduction="mean",
            )
        elif self.semantic_loss == "cosine":
            # Compute cosine similarity and transform to a loss value
            # Add small epsilon to avoid numerical issues
            loss = -torch.log(
                0.5
                + 1e-6
                - F.cosine_similarity(
                    semantic_prediction[:, :min_len], semantic[:, :min_len], dim=1
                )
                / 2
            ).mean()

        return loss

    def forward(
        self, x: torch.Tensor, use_dual_decoder: bool = False
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]
    ]:
        """DAC forward propagation.

        Args:
            x: Input tensor of shape (B, 1, T).
            use_dual_decoder: Whether to use dual decoder for encoder out.

        Returns:
            Tuple containing:
                - resynthesized audio
                - commitment loss
                - quantization loss
                - semantic loss
                - resynthesized audio from encoder (if dual decoder is used)
        """
        # Extract semantic features
        semantic = self._extract_semantic_features(x)

        # Encode input
        encoder_out = self.encoder(x)

        # Select target bandwidth
        bw_idx = random.randint(0, len(self.target_bandwidths) - 1)
        bw = self.target_bandwidths[bw_idx]

        # Apply quantization
        quantized_list, _, _, commit_loss, _ = self.quantizer(
            encoder_out, self.frame_rate, bw, return_list=True
        )

        # Extract semantic stream and final quantized output
        semantic_stream = quantized_list[0].permute(0, 2, 1)
        quantized = quantized_list[-1]

        # Calculate quantization loss
        quantization_loss = self.l1_quantization_loss(
            encoder_out, quantized.detach()
        ) + self.l2_quantization_loss(encoder_out, quantized.detach())

        # Generate semantic prediction and calculate loss
        semantic_prediction = self.semantic_prediction(semantic_stream)
        semantic_loss = self._calculate_semantic_loss(semantic_prediction, semantic)

        # Decode quantized representation
        resyn_audio = self.decoder(quantized)

        # Optionally decode directly from encoder output
        resyn_audio_real = None
        if use_dual_decoder:
            resyn_audio_real = self.decoder(encoder_out)

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
    ) -> torch.Tensor:
        """DAC codec encoding.

        Args:
            x: Input tensor of shape (B, 1, T) or (T,).
            target_bw: Target bandwidth.

        Returns:
            Neural codecs.
        """
        # Ensure input has correct dimensions
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        elif x.dim() == 2:
            x = x.unsqueeze(1)  # Add channel dimension

        # Run through encoder
        encoder_out = self.encoder(x.float())

        # Select bandwidth
        if target_bw is None:
            bw = self.target_bandwidths[-1]  # Use maximum bandwidth by default
        else:
            bw = target_bw

        # Encode to discrete codes
        codes = self.quantizer.encode(encoder_out, self.frame_rate, bw)

        return codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """DAC codec decoding.

        Args:
            codes: Neural codecs.

        Returns:
            Resynthesized audio.
        """
        # Convert codes back to continuous representation
        quantized = self.quantizer.decode(codes)

        # Decode to audio
        resyn_audio = self.decoder(quantized)

        return resyn_audio

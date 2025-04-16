# Copyright 2024 Yihan Wu
# Copyright 2025 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""DAC Modules - Refined Implementation."""
import copy
import functools
import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import InplaceFunction
from typeguard import typechecked

from espnet2.gan_codec.abs_gan_codec import AbsGANCodec
from espnet2.gan_codec.dac.dac import DACDiscriminator
from espnet2.gan_codec.shared.decoder.seanet import SEANetDecoder
from espnet2.gan_codec.shared.encoder.seanet import SEANetEncoder
from espnet2.gan_codec.shared.loss.freq_loss import MultiScaleMelSpectrogramLoss
from espnet2.gan_tts.hifigan.loss import (
    DiscriminatorAdversarialLoss,
    FeatureMatchLoss,
    GeneratorAdversarialLoss,
)
from espnet2.torch_utils.device_funcs import force_gatherable


class CustomRoundFunc(InplaceFunction):
    """Custom rounding function for the finite scalar quantization.

    Rounds input * factor and divides by factor - creating a quantization with steps of 1/factor.
    Maintains gradient flow during backpropagation using straight-through estimator.

    Attributes:
        factor (float): The quantization factor that determines step size (default: 3.0).
    """

    factor = 3.0  # Default factor

    @staticmethod
    def forward(ctx, input):
        """Forward pass of custom rounding function.

        Args:
            ctx: Context object for storing information for backward pass
            input (torch.Tensor): Input tensor to be quantized

        Returns:
            torch.Tensor: Quantized tensor (rounded to 1/factor precision)
        """
        # Store input for backward pass
        ctx.save_for_backward(input)

        # Apply quantization with current factor
        factor = CustomRoundFunc.factor
        return torch.round(factor * input) / factor

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass of custom rounding function using straight-through estimator.

        Args:
            ctx: Context object containing stored tensors
            grad_output (torch.Tensor): Gradient from subsequent layers

        Returns:
            torch.Tensor: Gradient propagated through the operation
        """
        # Straight-through estimator for gradient
        return grad_output.clone()

    @staticmethod
    def set_factor(new_factor: float) -> None:
        """Set quantization factor for all instances of this quantizer.

        Args:
            new_factor (float): New quantization factor. Higher values create finer quantization.

        Raises:
            ValueError: If quantization factor is not positive
        """
        if new_factor <= 0:
            raise ValueError("Quantization factor must be positive")
        CustomRoundFunc.factor = float(new_factor)

    @staticmethod
    def get_factor() -> float:
        """Get current quantization factor.

        Returns:
            float: Current quantization factor.
        """
        return CustomRoundFunc.factor


class FSQDAC(AbsGANCodec):
    """Finite Scalar Quantization Deep Audio Codec (FSQ-DAC) model.

    This class implements a neural audio codec that uses finite scalar quantization
    for compressing audio representations. The model consists of an encoder-decoder
    architecture with an adversarial discriminator for improved audio quality.
    """

    @typechecked
    def __init__(
        self,
        sampling_rate: int = 24000,
        generator_params: Dict[str, Any] = {
            "hidden_dim": 128,
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
            "decoder_trim_right_ratio": 1.0,
            "decoder_final_activation": None,
            "decoder_final_activation_params": None,
            "quantizer_codedim": 4,
            "quantizer_factor": 3.0,
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
        # Loss related parameters
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
        skip_quantizer_updates: int = 0,
        lambda_quantization: float = 1.0,
        lambda_reconstruct: float = 1.0,
        lambda_adv: float = 1.0,
        lambda_feat_match: float = 2.0,
        lambda_mel: float = 45.0,
        cache_generator_outputs: bool = False,
    ):
        """Initialize FSQDAC model.

        Args:
            sampling_rate (int): Audio sampling rate in Hz.
            generator_params (Dict[str, Any]): Parameters for the generator model.
            discriminator_params (Dict[str, Any]): Parameters for the discriminator model.
            generator_adv_loss_params (Dict[str, Any]): Parameters for generator adversarial loss.
            discriminator_adv_loss_params (Dict[str, Any]): Parameters for discriminator adversarial loss.
            use_feat_match_loss (bool): Whether to use feature matching loss.
            feat_match_loss_params (Dict[str, Any]): Parameters for feature matching loss.
            use_mel_loss (bool): Whether to use mel-spectrogram reconstruction loss.
            mel_loss_params (Dict[str, Any]): Parameters for mel-spectrogram loss.
            use_dual_decoder (bool): Whether to use dual decoder approach.
            skip_quantizer_updates (int): Number of updates to skip quantizer training.
            lambda_quantization (float): Weight for quantization loss.
            lambda_reconstruct (float): Weight for reconstruction loss.
            lambda_adv (float): Weight for adversarial loss.
            lambda_feat_match (float): Weight for feature matching loss.
            lambda_mel (float): Weight for mel-spectrogram loss.
            cache_generator_outputs (bool): Whether to cache generator outputs.
        """
        super().__init__()

        # Define modules
        generator_params = dict(
            generator_params
        )  # Create a copy to avoid modifying the original
        generator_params["sample_rate"] = sampling_rate
        self.generator = DACGenerator(**generator_params)
        self.discriminator = DACDiscriminator(**discriminator_params)

        # Define losses
        self.generator_adv_loss = GeneratorAdversarialLoss(**generator_adv_loss_params)
        self.discriminator_adv_loss = DiscriminatorAdversarialLoss(
            **discriminator_adv_loss_params
        )
        self.generator_reconstruct_loss = nn.L1Loss(reduction="mean")

        # Training configuration
        self.skip_quantizer_updates = skip_quantizer_updates
        self.register_buffer("num_updates", torch.zeros(1, dtype=torch.long))

        # Optional loss components
        self.use_feat_match_loss = use_feat_match_loss
        if self.use_feat_match_loss:
            self.feat_match_loss = FeatureMatchLoss(**feat_match_loss_params)

        self.use_mel_loss = use_mel_loss
        if self.use_mel_loss:
            mel_loss_params = dict(mel_loss_params)  # Create a copy
            mel_loss_params["fs"] = sampling_rate
            self.mel_loss = MultiScaleMelSpectrogramLoss(**mel_loss_params)

        self.use_dual_decoder = use_dual_decoder
        if self.use_dual_decoder and not self.use_mel_loss:
            raise ValueError("Dual decoder requires Mel loss to be enabled")

        # Loss coefficients
        self.lambda_quantization = lambda_quantization
        self.lambda_reconstruct = lambda_reconstruct
        self.lambda_adv = lambda_adv
        self.lambda_feat_match = lambda_feat_match if self.use_feat_match_loss else 0.0
        self.lambda_mel = lambda_mel if self.use_mel_loss else 0.0

        # Caching mechanism
        self.cache_generator_outputs = cache_generator_outputs
        self._cache = None

        # Store model metadata
        self.fs = sampling_rate
        self.num_streams = 1
        self.frame_shift = functools.reduce(
            lambda x, y: x * y, generator_params["encdec_ratios"]
        )
        self.code_size_per_stream = [
            generator_params["quantizer_factor"]
            ** generator_params["quantizer_codedim"]
        ]

    def meta_info(self) -> Dict[str, Any]:
        """Return metadata about the model.

        Returns:
            Dict[str, Any]: Dictionary containing metadata.
        """
        return {
            "fs": self.fs,
            "num_streams": self.num_streams,
            "frame_shift": self.frame_shift,
            "code_size_per_stream": self.code_size_per_stream,
            "quantizer_factor": CustomRoundFunc.get_factor(),
        }

    def forward(
        self,
        audio: torch.Tensor,
        forward_generator: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """Perform model forward pass.

        Args:
            audio (torch.Tensor): Audio waveform tensor (B, T_wav).
            forward_generator (bool): Whether to forward generator or discriminator.

        Returns:
            Dict[str, Any]:
                - loss (torch.Tensor): Loss scalar tensor.
                - stats (Dict[str, float]): Statistics to be monitored.
                - weight (torch.Tensor): Weight tensor to summarize losses.
                - optim_idx (int): Optimizer index (0 for G and 1 for D).
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
            audio (torch.Tensor): Audio waveform tensor (B, T_wav).

        Returns:
            Dict[str, Any]:
                - loss (torch.Tensor): Loss scalar tensor.
                - stats (Dict[str, float]): Statistics to be monitored.
                - weight (torch.Tensor): Weight tensor to summarize losses.
                - optim_idx (int): Optimizer index (0 for G).
        """
        # Setup
        batch_size = audio.size(0)
        audio = audio.unsqueeze(1)  # Add channel dimension (B, 1, T_wav)

        # Calculate generator outputs
        reuse_cache = self.cache_generator_outputs and self._cache is not None
        if not reuse_cache:
            audio_hat, quantization_loss, audio_hat_real = self.generator(
                audio, use_dual_decoder=self.use_dual_decoder
            )

            # Store cache if enabled during training
            if self.training and self.cache_generator_outputs:
                self._cache = (audio_hat, quantization_loss, audio_hat_real)
        else:
            audio_hat, quantization_loss, audio_hat_real = self._cache

        # Determine which audio reconstruction to use based on training phase
        is_quantizer_active = self.skip_quantizer_updates <= self.num_updates
        target_audio = (
            audio_hat
            if (is_quantizer_active and self.use_dual_decoder)
            else audio_hat_real
        )

        # Calculate discriminator outputs
        p_hat = self.discriminator(target_audio)
        with torch.no_grad():
            # Do not store discriminator gradient in generator turn
            p = self.discriminator(audio)

        # Calculate losses
        adv_loss = self.generator_adv_loss(p_hat) * self.lambda_adv
        reconstruct_loss = (
            self.generator_reconstruct_loss(audio, target_audio)
            * self.lambda_reconstruct
        )

        # Initialize total loss and statistics
        loss = adv_loss + reconstruct_loss
        stats = {
            "adv_loss": adv_loss.item(),
            "codec_quantization_loss": quantization_loss.item(),  # Just for reference, no gradient
            "reconstruct_loss": reconstruct_loss.item(),
        }

        # Add feature matching loss if enabled
        if self.use_feat_match_loss:
            feat_match_loss = self.feat_match_loss(p_hat, p) * self.lambda_feat_match
            loss = loss + feat_match_loss
            stats["feat_match_loss"] = feat_match_loss.item()

        # Add mel-spectrogram loss if enabled
        if self.use_mel_loss:
            # Ensure model is in the correct training phase if using the dual decoder
            if not self.use_dual_decoder and not is_quantizer_active:
                raise ValueError(
                    "Skip quantizer updates must be used with dual decoder"
                )

            # Mel loss for quantized reconstruction
            if is_quantizer_active:
                mel_loss = self.mel_loss(audio_hat, audio) * self.lambda_mel
                loss = loss + mel_loss
                stats["mel_loss"] = mel_loss.item()

            # Mel loss for direct reconstruction (used in early training or with dual decoder)
            if self.use_dual_decoder:
                mel_loss_real = self.mel_loss(audio_hat_real, audio) * self.lambda_mel
                loss = loss + mel_loss_real
                stats["mel_loss_real"] = mel_loss_real.item()
                # Use mel_loss_real as mel_loss if mel_loss not already set
                if "mel_loss" not in stats:
                    stats["mel_loss"] = mel_loss_real.item()

        # Increment update counter
        self.num_updates += 1

        # Add total loss to stats
        stats["loss"] = loss.item()

        # Make loss, stats, and weight gatherable across devices
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
            audio (torch.Tensor): Audio waveform tensor (B, T_wav).

        Returns:
            Dict[str, Any]:
                - loss (torch.Tensor): Loss scalar tensor.
                - stats (Dict[str, float]): Statistics to be monitored.
                - weight (torch.Tensor): Weight tensor to summarize losses.
                - optim_idx (int): Optimizer index (1 for D).
        """
        # Setup
        batch_size = audio.size(0)
        audio = audio.unsqueeze(1)  # Add channel dimension (B, 1, T_wav)

        # Calculate generator outputs
        reuse_cache = self.cache_generator_outputs and self._cache is not None
        if not reuse_cache:
            audio_hat, codec_quantization_loss, audio_hat_real = self.generator(
                audio, use_dual_decoder=self.use_dual_decoder
            )

            # Store cache if enabled
            if self.cache_generator_outputs:
                self._cache = (audio_hat, codec_quantization_loss, audio_hat_real)
        else:
            audio_hat, codec_quantization_loss, audio_hat_real = self._cache

        # Determine which audio reconstruction to use based on training phase
        is_quantizer_active = self.skip_quantizer_updates <= self.num_updates
        target_audio = (
            audio_hat
            if (is_quantizer_active and self.use_dual_decoder)
            else audio_hat_real
        )

        # Calculate discriminator outputs (detach generator outputs)
        p_hat = self.discriminator(target_audio.detach())
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

        # Make loss, stats, and weight gatherable across devices
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
        quantizer_factor: Optional[float] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Run inference.

        Args:
            x (torch.Tensor): Input audio (T_wav,).
            quantizer_factor (Optional[float]): Override the default quantization factor.
            **kwargs: Additional keyword arguments.

        Returns:
            Dict[str, torch.Tensor]:
                * wav (torch.Tensor): Generated waveform tensor (T_wav,).
                * codec (torch.Tensor): Generated neural codec (T_code, N_stream).
        """
        # Add batch dimension if necessary
        if x.dim() == 1:
            x = x.unsqueeze(0)  # (1, T_wav)

        # Add channel dimension if necessary
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, T_wav)

        codec = self.generator.encode(x, quantizer_factor=quantizer_factor)
        wav = self.generator.decode(codec, quantizer_factor=quantizer_factor)

        # Remove batch dimension if it was added
        if x.size(0) == 1:
            wav = wav.squeeze(0)
            codec = codec.squeeze(0)

        return {"wav": wav, "codec": codec}

    def encode(
        self,
        x: torch.Tensor,
        quantizer_factor: Optional[float] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Run encoding.

        Args:
            x (torch.Tensor): Input audio (T_wav,).
            quantizer_factor (Optional[float]): Override the default quantization factor.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: Generated codes (T_code, N_stream).
        """
        # Handle various input shapes consistently
        input_dim = x.dim()

        # Add batch dimension if necessary
        if input_dim == 1:
            x = x.unsqueeze(0)  # (1, T_wav)

        # Add channel dimension if necessary
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, T_wav)

        # Encode
        codes = self.generator.encode(x, quantizer_factor=quantizer_factor)

        # Remove batch dimension if it was added
        if input_dim == 1 and codes.size(0) == 1:
            codes = codes.squeeze(0)

        return codes

    def decode(
        self,
        x: torch.Tensor,
        quantizer_factor: Optional[float] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Run decoding.

        Args:
            x (torch.Tensor): Input codes (T_code, N_stream) or (B, T_code, N_stream).
            quantizer_factor (Optional[float]): Override the default quantization factor.
                Must match the factor used during encoding.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: Generated waveform (T_wav,) or (B, T_wav).
        """
        # Handle various input shapes consistently
        input_dim = x.dim()

        # Add batch dimension if necessary
        if input_dim == 2:  # (T_code, N_stream)
            x = x.unsqueeze(0)  # (1, T_code, N_stream)

        # Decode
        wav = self.generator.decode(x, quantizer_factor=quantizer_factor)

        # Remove batch dimension if it was added
        if input_dim == 2 and wav.size(0) == 1:
            wav = wav.squeeze(0)

        return wav


class DACGenerator(nn.Module):
    """DAC generator module with encoder, quantizer, and decoder components."""

    @typechecked
    def __init__(
        self,
        sample_rate: int = 24000,
        hidden_dim: int = 128,
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
        decoder_final_activation_params: Optional[Dict[str, Any]] = None,
        quantizer_codedim: int = 4,
        quantizer_factor: float = 3.0,
    ):
        """Initialize DAC Generator.

        Args:
            sample_rate (int): Audio sampling rate in Hz.
            hidden_dim (int): Hidden dimension size.
            codebook_dim (int): Dimension of codebook entries.
            encdec_channels (int): Number of input/output audio channels.
            encdec_n_filters (int): Base number of convolutional filters.
            encdec_n_residual_layers (int): Number of residual layers.
            encdec_ratios (List[int]): Upsampling/downsampling ratios.
            encdec_activation (str): Activation function name.
            encdec_activation_params (Dict[str, Any]): Activation function parameters.
            encdec_norm (str): Normalization method.
            encdec_norm_params (Dict[str, Any]): Normalization parameters.
            encdec_kernel_size (int): Kernel size for convolutions.
            encdec_residual_kernel_size (int): Kernel size for residual convolutions.
            encdec_last_kernel_size (int): Kernel size for last layer.
            encdec_dilation_base (int): Base for dilation exponential.
            encdec_causal (bool): Whether to use causal convolutions.
            encdec_pad_mode (str): Padding mode for convolutions.
            encdec_true_skip (bool): Whether to use true skip connections.
            encdec_compress (int): Compression factor for channels.
            encdec_lstm (int): Number of LSTM layers.
            decoder_trim_right_ratio (float): Ratio for trimming right side in decoder.
            decoder_final_activation (Optional[str]): Final activation function.
            decoder_final_activation_params (Optional[Dict]): Final activation parameters.
            quantizer_codedim (int): Code dimensions for the quantizer.
            quantizer_factor (float): Quantizer factor.
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
        )

        # Initialize quantizer components
        self.quantizer_pre = nn.Linear(hidden_dim, quantizer_codedim)
        self.quantizer = CustomRoundFunc
        self.quantizer_after = nn.Linear(quantizer_codedim, hidden_dim)

        # Setup model parameters
        self.sample_rate = sample_rate
        self.frame_rate = math.ceil(sample_rate / np.prod(encdec_ratios))

        # Set quantization factor
        self.quantizer_factor = quantizer_factor
        CustomRoundFunc.set_factor(quantizer_factor)

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

    def forward(
        self, x: torch.Tensor, use_dual_decoder: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """DAC generator forward propagation.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 1, T).
            use_dual_decoder (bool): Whether to use dual decoder for encoder out.

        Returns:
            Tuple containing:
                - torch.Tensor: Resynthesized audio from quantized features.
                - torch.Tensor: Quantization loss.
                - Optional[torch.Tensor]: Resynthesized audio directly from encoder (if use_dual_decoder=True).
        """
        # Encode input
        encoder_out = self.encoder(x)

        # Apply quantization
        quantizer_input = self.quantizer_pre(encoder_out.permute(0, 2, 1))
        quantized = self.quantizer.apply(quantizer_input)
        quantized = self.quantizer_after(quantized)
        quantized = quantized.permute(0, 2, 1)

        # Calculate quantization loss
        quantization_loss = F.l1_loss(quantized, encoder_out)

        # Decode quantized features
        resyn_audio = self.decoder(quantized)

        # Optionally decode directly from encoder features (for dual decoder approach)
        resyn_audio_real = self.decoder(encoder_out) if use_dual_decoder else None

        return resyn_audio, quantization_loss, resyn_audio_real

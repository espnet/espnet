# Copyright 2024 Yihan Wu
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""DAC Modules."""
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
from espnet2.gan_codec.shared.encoder.seanet import SEANetEncoder
from espnet2.gan_codec.shared.loss.freq_loss import MultiScaleMelSpectrogramLoss
from espnet2.gan_codec.shared.quantizer.residual_vq import ResidualVectorQuantizer
from espnet2.gan_tts.hifigan.loss import (
    DiscriminatorAdversarialLoss,
    FeatureMatchLoss,
    GeneratorAdversarialLoss,
)
from espnet2.torch_utils.device_funcs import force_gatherable


class DAC(AbsGANCodec):
    """
    DAC model for audio processing using GAN architecture.

    The DAC (Discrete Audio Codec) model utilizes a GAN-based architecture to
    encode and decode audio signals. It features a generator that processes audio
    waveforms and a discriminator that assesses the quality of generated outputs.
    This model supports various loss functions and can be configured with
    multiple parameters for fine-tuning.

    Attributes:
        generator (DACGenerator): The generator module responsible for audio
            synthesis.
        discriminator (DACDiscriminator): The discriminator module that evaluates
            the generated audio.
        generator_adv_loss (GeneratorAdversarialLoss): Loss function for the
            generator.
        generator_reconstruct_loss (torch.nn.L1Loss): Loss function for audio
            reconstruction.
        discriminator_adv_loss (DiscriminatorAdversarialLoss): Loss function for
            the discriminator.
        use_feat_match_loss (bool): Flag to enable feature matching loss.
        feat_match_loss (FeatureMatchLoss): Loss function for feature matching.
        use_mel_loss (bool): Flag to enable mel spectrogram loss.
        mel_loss (MultiScaleMelSpectrogramLoss): Loss function for mel
            spectrograms.
        use_dual_decoder (bool): Flag to use dual decoding.
        cache_generator_outputs (bool): Flag to cache generator outputs.
        fs (int): Sampling rate of the audio.
        num_streams (int): Number of quantization streams.
        frame_shift (int): Frame shift size.
        code_size_per_stream (List[int]): Code size per quantization stream.

    Args:
        sampling_rate (int): The sampling rate of the audio (default: 24000).
        generator_params (Dict[str, Any]): Parameters for the generator model.
        discriminator_params (Dict[str, Any]): Parameters for the discriminator model.
        generator_adv_loss_params (Dict[str, Any]): Parameters for generator
            adversarial loss.
        discriminator_adv_loss_params (Dict[str, Any]): Parameters for
            discriminator adversarial loss.
        use_feat_match_loss (bool): Whether to use feature matching loss
            (default: True).
        feat_match_loss_params (Dict[str, Any]): Parameters for feature matching loss.
        use_mel_loss (bool): Whether to use mel loss (default: True).
        mel_loss_params (Dict[str, Any]): Parameters for mel loss.
        use_dual_decoder (bool): Whether to use a dual decoder (default: True).
        lambda_quantization (float): Weight for quantization loss (default: 1.0).
        lambda_reconstruct (float): Weight for reconstruction loss (default: 1.0).
        lambda_commit (float): Weight for commitment loss (default: 1.0).
        lambda_adv (float): Weight for adversarial loss (default: 1.0).
        lambda_feat_match (float): Weight for feature matching loss (default: 2.0).
        lambda_mel (float): Weight for mel loss (default: 45.0).
        cache_generator_outputs (bool): Whether to cache generator outputs
            (default: False).

    Examples:
        >>> dac_model = DAC(sampling_rate=22050)
        >>> audio_input = torch.randn(1, 22050)  # Simulated audio input
        >>> output = dac_model(audio_input)
        >>> print(output["loss"])

    Note:
        The DAC model requires proper configuration of the generator and
        discriminator parameters to function effectively. Make sure to consult
        the documentation for detailed descriptions of each parameter.

    Raises:
        AssertionError: If dual decoder is used without enabling mel loss.
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
        cache_generator_outputs: bool = False,
    ):
        """Intialize DAC model.

        Args:
             TODO(jiatong)
        """
        super().__init__()

        # define modules
        generator_params.update(sample_rate=sampling_rate)
        self.generator = DACGenerator(**generator_params)
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
            lambda x, y: x * y, generator_params["encdec_ratios"]
        )
        self.code_size_per_stream = [
            generator_params["quantizer_bins"]
        ] * self.num_streams

    def meta_info(self) -> Dict[str, Any]:
        """
        Retrieve metadata information of the DAC model.

        This method returns a dictionary containing key metadata attributes
        of the DAC model, which includes the sampling frequency, number of
        streams, frame shift, and the code size per stream.

        Returns:
            Dict[str, Any]: A dictionary containing the following keys:
                - fs (int): The sampling frequency of the model.
                - num_streams (int): The number of quantizer streams.
                - frame_shift (int): The frame shift calculated from
                  the encoder-decoder ratios.
                - code_size_per_stream (List[int]): A list indicating the
                  code size for each stream.

        Examples:
            >>> dac_model = DAC()
            >>> info = dac_model.meta_info()
            >>> print(info)
            {'fs': 24000, 'num_streams': 8, 'frame_shift': 640,
             'code_size_per_stream': [1024, 1024, 1024, 1024, 1024,
             1024, 1024, 1024]}
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
        """
        Perform generator forward.

        Args:
            audio (Tensor): Audio waveform tensor (B, T_wav).
            forward_generator (bool): Whether to forward generator.

        Returns:
            Dict[str, Any]:
                - loss (Tensor): Loss scalar tensor.
                - stats (Dict[str, float]): Statistics to be monitored.
                - weight (Tensor): Weight tensor to summarize losses.
                - optim_idx (int): Optimizer index (0 for G and 1 for D).

        Examples:
            >>> model = DAC()
            >>> audio_input = torch.randn(1, 16000)  # Example audio tensor
            >>> output = model.forward(audio_input)
            >>> print(output.keys())
            dict_keys(['loss', 'stats', 'weight', 'optim_idx'])
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
        """
        Run inference to generate audio from input.

        This method takes an input audio tensor and generates the corresponding
        output waveform and neural codec. The input tensor should be of shape
        (T_wav,) where T_wav is the length of the audio waveform.

        Args:
            x (Tensor): Input audio tensor of shape (T_wav,).

        Returns:
            Dict[str, Tensor]:
                - wav (Tensor): Generated waveform tensor of shape (T_wav,).
                - codec (Tensor): Generated neural codec tensor of shape
                  (T_code, N_stream).

        Examples:
            >>> model = DAC()
            >>> input_audio = torch.randn(24000)  # Example audio input
            >>> output = model.inference(input_audio)
            >>> generated_wav = output['wav']
            >>> generated_codec = output['codec']

        Note:
            Ensure that the input tensor is appropriately preprocessed and
            matches the expected input format of the model.
        """
        codec = self.generator.encode(x)
        wav = self.generator.decode(codec)

        return {"wav": wav, "codec": codec}

    def encode(
        self,
        x: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Run encoding.

        This method encodes the input audio tensor into a set of generated codes
        using the DAC generator. The encoding process involves passing the audio
        waveform through the generator's encoder and quantizer.

        Args:
            x (Tensor): Input audio (T_wav,). The shape of the tensor should
            be compatible with the expected input of the encoder.

        Returns:
            Tensor: Generated codes (T_code, N_stream). The output tensor contains
            the encoded representation of the input audio, where T_code is the
            length of the generated codes and N_stream is the number of
            quantization streams.

        Examples:
            >>> model = DAC()
            >>> audio_input = torch.randn(1, 24000)  # Simulated audio input
            >>> encoded_codes = model.encode(audio_input)
            >>> print(encoded_codes.shape)
            torch.Size([T_code, N_stream])  # Shape depends on the input and model params
        """
        return self.generator.encode(x)

    def decode(
        self,
        x: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Run decoding.

        This method takes the input codes and generates the corresponding
        waveform using the DAC generator.

        Args:
            x (Tensor): Input codes (T_code, N_stream).

        Returns:
            Tensor: Generated waveform (T_wav,).

        Examples:
            >>> dac_model = DAC()
            >>> codes = torch.randn(100, 8)  # Example input codes
            >>> waveform = dac_model.decode(codes)
            >>> print(waveform.shape)
            torch.Size([T_wav,])  # Output shape will depend on the model

        Note:
            Ensure that the input tensor `x` has the correct shape,
            matching the expected dimensions for the decoder.
        """
        return self.generator.decode(x)


class DACGenerator(nn.Module):
    """
    DAC generator module.

    This module implements the generator for the DAC (Discrete
    Audio Codec) model. It utilizes an encoder-decoder architecture
    with quantization to generate audio waveforms from input tensors.
    The generator is designed to be flexible, allowing for various
    configurations of encoder and decoder parameters.

    Attributes:
        encoder (SEANetEncoder): The encoder component of the DAC generator.
        quantizer (ResidualVectorQuantizer): The quantizer for encoding.
        target_bandwidths (List[float]): List of target bandwidths for
            quantization.
        sample_rate (int): The sample rate of the audio.
        frame_rate (int): The frame rate calculated from the sample rate
            and encoder-decoder ratios.
        decoder (SEANetDecoder): The decoder component of the DAC generator.
        l1_quantization_loss (torch.nn.L1Loss): L1 loss for quantization.
        l2_quantization_loss (torch.nn.MSELoss): L2 loss for quantization.

    Args:
        sample_rate (int): The sample rate of the audio (default: 24000).
        hidden_dim (int): Dimension of hidden layers (default: 128).
        codebook_dim (int): Dimension of the codebook for quantization
            (default: 8).
        encdec_channels (int): Number of channels for encoder/decoder
            (default: 1).
        encdec_n_filters (int): Number of filters for encoder/decoder
            (default: 32).
        encdec_n_residual_layers (int): Number of residual layers
            (default: 1).
        encdec_ratios (List[int]): Ratios for downsampling (default:
            [8, 5, 4, 2]).
        encdec_activation (str): Activation function used (default: "Snake").
        encdec_activation_params (Dict[str, Any]): Parameters for
            activation function (default: {}).
        encdec_norm (str): Normalization method used (default:
            "weight_norm").
        encdec_norm_params (Dict[str, Any]): Parameters for normalization
            (default: {}).
        encdec_kernel_size (int): Kernel size for convolution layers
            (default: 7).
        encdec_residual_kernel_size (int): Kernel size for residual
            connections (default: 7).
        encdec_last_kernel_size (int): Kernel size for the last layer
            (default: 7).
        encdec_dilation_base (int): Dilation base for convolution layers
            (default: 2).
        encdec_causal (bool): Whether to use causal convolutions (default:
            False).
        encdec_pad_mode (str): Padding mode for convolutions (default:
            "reflect").
        encdec_true_skip (bool): Whether to use true skip connections
            (default: False).
        encdec_compress (int): Compression factor for the encoder (default:
            2).
        encdec_lstm (int): Number of LSTM layers (default: 2).
        decoder_trim_right_ratio (float): Trim ratio for the decoder output
            (default: 1.0).
        decoder_final_activation (Optional[str]): Final activation function
            for the decoder (default: None).
        decoder_final_activation_params (Optional[dict]): Parameters for
            the final activation function (default: None).
        quantizer_n_q (int): Number of quantizers (default: 8).
        quantizer_bins (int): Number of bins for quantization (default:
            1024).
        quantizer_decay (float): Decay factor for quantization (default:
            0.99).
        quantizer_kmeans_init (bool): Whether to initialize with K-means
            (default: True).
        quantizer_kmeans_iters (int): Number of K-means iterations
            (default: 50).
        quantizer_threshold_ema_dead_code (int): Threshold for dead code
            (default: 2).
        quantizer_target_bandwidth (List[float]): Target bandwidths for
            quantization (default: [7.5, 15]).
        quantizer_dropout (bool): Whether to use dropout in the quantizer
            (default: True).

    Examples:
        >>> generator = DACGenerator(sample_rate=22050, hidden_dim=256)
        >>> input_tensor = torch.randn(1, 1, 48000)  # (B, C, T)
        >>> output, commit_loss, quantization_loss, audio_hat_real = generator(input_tensor)
    """

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

        # quantization loss
        self.l1_quantization_loss = torch.nn.L1Loss(reduction="mean")
        self.l2_quantization_loss = torch.nn.MSELoss(reduction="mean")

    def forward(self, x: torch.Tensor, use_dual_decoder: bool = False):
        """
        Perform generator forward.

        Args:
            audio (Tensor): Audio waveform tensor (B, T_wav).
            forward_generator (bool): Whether to forward generator.

        Returns:
            Dict[str, Any]:
                - loss (Tensor): Loss scalar tensor.
                - stats (Dict[str, float]): Statistics to be monitored.
                - weight (Tensor): Weight tensor to summarize losses.
                - optim_idx (int): Optimizer index (0 for G and 1 for D).

        Examples:
            >>> model = DAC()
            >>> audio_input = torch.randn(1, 24000)  # Example audio input
            >>> output = model.forward(audio_input, forward_generator=True)
            >>> print(output['loss'].item())
        """
        encoder_out = self.encoder(x)
        max_idx = len(self.target_bandwidths) - 1

        # randomly pick up one bandwidth
        bw = self.target_bandwidths[random.randint(0, max_idx)]

        # Forward quantizer
        quantized, _, _, commit_loss, quantization_loss = self.quantizer(
            encoder_out, self.frame_rate, bw
        )

        resyn_audio = self.decoder(quantized)

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
        """
        Run encoding.

        Args:
            x (Tensor): Input audio (T_wav,).

        Returns:
            Tensor: Generated codes (T_code, N_stream).

        Examples:
            >>> model = DAC()
            >>> audio_input = torch.randn(1, 24000)  # Simulate 1 second of audio
            >>> codes = model.encode(audio_input)
            >>> print(codes.shape)  # Should output the shape of generated codes

        Note:
            The input tensor `x` should be a 1D tensor representing audio
            waveform data with a shape of (T_wav,). The output will be a
            tensor containing the generated codes with shape (T_code, N_stream).
        """

        encoder_out = self.encoder(x)
        if target_bw is None:
            bw = self.target_bandwidths[-1]
        else:
            bw = target_bw
        codes = self.quantizer.encode(encoder_out, self.frame_rate, bw)
        return codes

    def decode(self, codes: torch.Tensor):
        """
        Run decoding to generate waveform from codes.

        This method takes in input codes, which are the output of the
        encoding process, and generates the corresponding waveform.

        Args:
            x (Tensor): Input codes (T_code, N_stream), where T_code is
                the length of the code sequence and N_stream is the number
                of quantized streams.

        Returns:
            Tensor: Generated waveform (T_wav,), which represents the
            reconstructed audio signal.

        Examples:
            >>> # Assume `codes` is a tensor containing encoded audio
            >>> generated_waveform = dac.decode(codes)
            >>> print(generated_waveform.shape)  # Output: (T_wav,)

        Note:
            The input codes should be properly formatted as per the model's
            specifications to ensure correct waveform generation.
        """
        quantized = self.quantizer.decode(codes)
        resyn_audio = self.decoder(quantized)
        return resyn_audio


class DACDiscriminator(nn.Module):
    """
    DAC discriminator module.

    This class implements the DAC Discriminator, which is responsible for
    distinguishing between real and generated audio signals. It utilizes a
    MultiScaleMultiPeriodMultiBand Discriminator architecture to process
    audio inputs at various scales and periods.

    Args:
        msmpmb_discriminator_params (Dict[str, Any]): Parameters for the
            MultiScaleMultiPeriodMultiBandDiscriminator. This includes
            settings for rates, periods, FFT sizes, and other relevant
            configurations for the period and band discriminators.
        scale_follow_official_norm (bool): If True, applies official
            normalization scale during the discriminator's processing.

    Attributes:
        msmpmb_discriminator (MultiScaleMultiPeriodMultiBandDiscriminator):
            An instance of the MultiScaleMultiPeriodMultiBandDiscriminator
            configured with the provided parameters.

    Returns:
        List[List[torch.Tensor]]: The output of the discriminator, which is
        a list of lists containing the outputs from each layer of the
        discriminator. Each list corresponds to a specific scale and period
        output.

    Examples:
        >>> discriminator = DACDiscriminator()
        >>> input_tensor = torch.randn(1, 1, 1024)  # Example input
        >>> outputs = discriminator(input_tensor)
        >>> print(len(outputs))  # Output length corresponds to number of scales

    Note:
        The DAC Discriminator is a critical component in the DAC model's
        adversarial training process, enabling the generator to learn
        more effectively by providing feedback on the quality of generated
        audio.
    """

    def __init__(
        self,
        # MultiScaleMultiPeriodMultiBandDiscriminator parameters
        msmpmb_discriminator_params: Dict[str, Any] = {
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
        scale_follow_official_norm: bool = False,
    ):
        """Initialize DAC Discriminator module.

        Args:

        """
        super().__init__()

        self.msmpmb_discriminator = MultiScaleMultiPeriodMultiBandDiscriminator(
            **msmpmb_discriminator_params
        )

    def forward(self, x: torch.Tensor) -> List[List[torch.Tensor]]:
        """
        Perform generator or discriminator forward pass.

        This method directs the input audio tensor to either the generator or
        the discriminator based on the `forward_generator` flag. If set to
        `True`, it forwards the audio to the generator, otherwise to the
        discriminator.

        Args:
            audio (Tensor): Audio waveform tensor of shape (B, T_wav), where
                B is the batch size and T_wav is the number of time steps.
            forward_generator (bool): Flag indicating whether to forward the
                audio through the generator (True) or the discriminator
                (False).

        Returns:
            Dict[str, Any]: A dictionary containing the following keys:
                - loss (Tensor): Loss scalar tensor computed during the forward
                  pass.
                - stats (Dict[str, float]): Statistics computed during the
                  forward pass for monitoring.
                - weight (Tensor): Weight tensor used to summarize losses.
                - optim_idx (int): Index indicating which optimizer to use
                  (0 for generator and 1 for discriminator).

        Examples:
            >>> model = DAC()
            >>> audio_input = torch.randn(8, 16000)  # Batch of 8, 16000 samples
            >>> output = model.forward(audio_input, forward_generator=True)
            >>> print(output.keys())
            dict_keys(['loss', 'stats', 'weight', 'optim_idx'])

        Note:
            This method will internally call either `_forward_generator` or
            `_forward_discrminator` based on the value of `forward_generator`.
        """
        msmpmb_outs = self.msmpmb_discriminator(x)
        return msmpmb_outs

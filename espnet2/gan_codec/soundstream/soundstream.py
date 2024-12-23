# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""SoundStream Modules."""
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
from espnet2.gan_codec.shared.discriminator.stft_discriminator import (
    ComplexSTFTDiscriminator,
)
from espnet2.gan_codec.shared.encoder.seanet import SEANetEncoder
from espnet2.gan_codec.shared.loss.freq_loss import MultiScaleMelSpectrogramLoss
from espnet2.gan_codec.shared.loss.loss_balancer import Balancer
from espnet2.gan_codec.shared.quantizer.residual_vq import ResidualVectorQuantizer
from espnet2.gan_tts.hifigan.hifigan import HiFiGANMultiScaleDiscriminator
from espnet2.gan_tts.hifigan.loss import (
    DiscriminatorAdversarialLoss,
    FeatureMatchLoss,
    GeneratorAdversarialLoss,
)
from espnet2.torch_utils.device_funcs import force_gatherable


class SoundStream(AbsGANCodec):
    """
    SoundStream model for audio generation and processing.

    This class implements the SoundStream model, which is a generative model
    for audio processing. It includes a generator and a discriminator, both
    of which are designed to work with audio waveforms. The model can perform
    tasks such as encoding, decoding, and generating audio waveforms.

    Attributes:
        generator (SoundStreamGenerator): The generator component of the model.
        discriminator (SoundStreamDiscriminator): The discriminator component of the model.
        generator_adv_loss (GeneratorAdversarialLoss): Adversarial loss for the generator.
        generator_reconstruct_loss (torch.nn.L1Loss): Reconstruction loss for the generator.
        discriminator_adv_loss (DiscriminatorAdversarialLoss): Adversarial loss for the discriminator.
        use_feat_match_loss (bool): Flag indicating whether to use feature matching loss.
        feat_match_loss (FeatureMatchLoss): Feature matching loss module.
        use_mel_loss (bool): Flag indicating whether to use mel loss.
        mel_loss (MultiScaleMelSpectrogramLoss): Mel spectrogram loss module.
        cache_generator_outputs (bool): Flag indicating whether to cache generator outputs.
        fs (int): Sampling rate for saving audio files.
        num_streams (int): Number of quantization streams.
        frame_shift (int): Frame shift size.
        code_size_per_stream (List[int]): Size of codes per quantization stream.
        loss_balancer (Optional[Balancer]): Loss balancer for handling multiple losses.

    Args:
        sampling_rate (int): Sampling rate for audio processing. Default is 24000.
        generator_params (Dict[str, Any]): Parameters for the generator.
        discriminator_params (Dict[str, Any]): Parameters for the discriminator.
        generator_adv_loss_params (Dict[str, Any]): Parameters for generator adversarial loss.
        discriminator_adv_loss_params (Dict[str, Any]): Parameters for discriminator adversarial loss.
        use_feat_match_loss (bool): Flag to use feature matching loss. Default is True.
        feat_match_loss_params (Dict[str, Any]): Parameters for feature matching loss.
        use_mel_loss (bool): Flag to use mel loss. Default is True.
        mel_loss_params (Dict[str, Any]): Parameters for mel loss.
        use_dual_decoder (bool): Flag to use dual decoder. Default is True.
        lambda_quantization (float): Weight for quantization loss. Default is 1.0.
        lambda_reconstruct (float): Weight for reconstruction loss. Default is 1.0.
        lambda_commit (float): Weight for commitment loss. Default is 1.0.
        lambda_adv (float): Weight for adversarial loss. Default is 1.0.
        lambda_feat_match (float): Weight for feature matching loss. Default is 2.0.
        lambda_mel (float): Weight for mel loss. Default is 45.0.
        cache_generator_outputs (bool): Flag to cache generator outputs. Default is False.
        use_loss_balancer (bool): Flag to use loss balancer. Default is False.
        balance_ema_decay (float): Exponential moving average decay for balancing losses. Default is 0.99.

    Examples:
        # Initialize the SoundStream model
        sound_stream = SoundStream(
            sampling_rate=24000,
            generator_params={"hidden_dim": 128, ...},  # Fill with actual params
            discriminator_params={"scales": 3, ...},  # Fill with actual params
        )

        # Perform forward pass with audio input
        output = sound_stream.forward(audio_input)

    Note:
        The model is designed to be used in a training loop where the generator and
        discriminator are optimized iteratively.

    Todo:
        - Complete the docstring with detailed descriptions for each argument.
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
        use_dual_decoder: bool = True,
        lambda_quantization: float = 1.0,
        lambda_reconstruct: float = 1.0,
        lambda_commit: float = 1.0,
        lambda_adv: float = 1.0,
        lambda_feat_match: float = 2.0,
        lambda_mel: float = 45.0,
        cache_generator_outputs: bool = False,
        use_loss_balancer: bool = False,
        balance_ema_decay: float = 0.99,
    ):
        """Intialize SoundStream model.

        Args:
             TODO(jiatong)
        """
        super().__init__()

        # define modules
        generator_params.update(sample_rate=sampling_rate)
        self.generator = SoundStreamGenerator(**generator_params)
        self.discriminator = SoundStreamDiscriminator(**discriminator_params)
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

        # loss balancer
        if use_loss_balancer:
            self.loss_balancer = Balancer(
                ema_decay=balance_ema_decay,
                per_batch_item=True,
            )
        else:
            self.loss_balancer = None

    def meta_info(self) -> Dict[str, Any]:
        """
        Retrieve meta information about the SoundStream model.

        This method provides key details about the model's configuration,
        including the sampling rate, number of streams, frame shift, and
        code size per stream.

        Returns:
            Dict[str, Any]: A dictionary containing the following key-value pairs:
                - fs (int): The sampling rate of the model.
                - num_streams (int): The number of quantization streams.
                - frame_shift (int): The frame shift size used in processing.
                - code_size_per_stream (List[int]): A list indicating the code
                  size for each stream.

        Examples:
            >>> sound_stream = SoundStream()
            >>> meta_info = sound_stream.meta_info()
            >>> print(meta_info)
            {'fs': 24000, 'num_streams': 8, 'frame_shift': 128,
             'code_size_per_stream': [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024]}
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

        This method computes the forward pass for either the generator or the
        discriminator, depending on the `forward_generator` flag.

        Args:
            audio (Tensor): Audio waveform tensor of shape (B, T_wav).
            forward_generator (bool): Flag indicating whether to forward the
                generator (True) or the discriminator (False).

        Returns:
            Dict[str, Any]: A dictionary containing the following keys:
                - loss (Tensor): Scalar tensor representing the total loss.
                - stats (Dict[str, float]): Statistics to be monitored,
                  including various loss components.
                - weight (Tensor): Weight tensor summarizing the losses.
                - optim_idx (int): Index indicating which optimizer to use
                  (0 for generator and 1 for discriminator).

        Examples:
            >>> audio_input = torch.randn(8, 24000)  # Batch of 8 audio samples
            >>> model = SoundStream()
            >>> output = model.forward(audio_input, forward_generator=True)
            >>> print(output.keys())
            dict_keys(['loss', 'stats', 'weight', 'optim_idx'])

        Note:
            The audio input should be pre-processed as necessary to fit the
            expected input shape.
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
        """
        Run inference on input audio.

        This method takes an input audio tensor, encodes it to a neural codec,
        and then decodes it back to a waveform. It is designed to be used for
        generating audio samples after the model has been trained.

        Args:
            x (Tensor): Input audio tensor of shape (T_wav,).

        Returns:
            Dict[str, Tensor]:
                * wav (Tensor): Generated waveform tensor of shape (T_wav,).
                * codec (Tensor): Generated neural codec of shape (T_code, N_stream).

        Examples:
            >>> model = SoundStream()
            >>> input_audio = torch.randn(24000)  # Example input (1 second of audio)
            >>> output = model.inference(input_audio)
            >>> generated_wav = output['wav']
            >>> generated_codec = output['codec']

        Note:
            The input audio tensor should be of shape (T_wav,) where T_wav is
            the length of the audio signal. The output includes both the
            generated waveform and the codec representation.
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

        This method processes the input audio tensor through the generator's
        encoder to produce a set of neural codes.

        Args:
            x (Tensor): Input audio tensor of shape (T_wav,).

        Returns:
            Tensor: Generated codes of shape (T_code, N_stream), where T_code
            is the length of the generated codes and N_stream is the number of
            quantization streams.

        Examples:
            >>> model = SoundStream()
            >>> input_audio = torch.randn(1, 24000)  # Example audio tensor
            >>> codes = model.encode(input_audio)
            >>> print(codes.shape)  # Output shape will be (T_code, N_stream)
        """
        return self.generator.encode(x)

    def decode(
        self,
        x: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Run decoding.

        This method takes encoded input codes and generates a waveform
        tensor by passing the codes through the generator's decoder.

        Args:
            x (Tensor): Input codes (T_code, N_stream).

        Returns:
            Tensor: Generated waveform (T_wav,).

        Examples:
            >>> model = SoundStream()
            >>> codes = torch.randn(100, 8)  # Example input codes
            >>> waveform = model.decode(codes)
            >>> print(waveform.shape)
            torch.Size([T_wav,])  # Shape of the generated waveform tensor

        Note:
            Ensure that the input tensor 'x' has the correct shape and
            data type expected by the generator's decoder.
        """
        return self.generator.decode(x)


class SoundStreamGenerator(nn.Module):
    """
    SoundStream generator module.

    This module implements the generator part of the SoundStream model, which is
    responsible for encoding and decoding audio signals. The generator utilizes
    a neural network architecture consisting of an encoder, quantizer, and decoder
    to perform the audio processing tasks.

    Attributes:
        encoder (SEANetEncoder): The encoder module that processes the input audio.
        quantizer (ResidualVectorQuantizer): The quantization module that encodes the
            features into a discrete representation.
        target_bandwidths (List[float]): List of target bandwidths for the quantizer.
        sample_rate (int): The sample rate of the audio signals.
        frame_rate (int): The frame rate calculated from the sample rate and encoding
            ratios.
        decoder (SEANetDecoder): The decoder module that reconstructs the audio from
            the quantized representation.
        l1_quantization_loss (torch.nn.L1Loss): Loss function for L1 quantization loss.
        l2_quantization_loss (torch.nn.MSELoss): Loss function for L2 quantization loss.

    Args:
        sample_rate (int): The sample rate of the audio (default: 24000).
        hidden_dim (int): The dimension of hidden layers (default: 128).
        encdec_channels (int): Number of channels for encoder/decoder (default: 1).
        encdec_n_filters (int): Number of filters in encoder/decoder (default: 32).
        encdec_n_residual_layers (int): Number of residual layers (default: 1).
        encdec_ratios (List[int]): Ratios for the encoder/decoder (default: [8, 5, 4, 2]).
        encdec_activation (str): Activation function used (default: "ELU").
        encdec_activation_params (Dict[str, Any]): Parameters for activation function
            (default: {"alpha": 1.0}).
        encdec_norm (str): Normalization type (default: "weight_norm").
        encdec_norm_params (Dict[str, Any]): Parameters for normalization (default: {}).
        encdec_kernel_size (int): Kernel size for the encoder/decoder (default: 7).
        encdec_residual_kernel_size (int): Kernel size for residual layers (default: 7).
        encdec_last_kernel_size (int): Kernel size for the last layer (default: 7).
        encdec_dilation_base (int): Dilation base for the encoder/decoder (default: 2).
        encdec_causal (bool): Whether to use causal convolution (default: False).
        encdec_pad_mode (str): Padding mode for convolution (default: "reflect").
        encdec_true_skip (bool): Whether to use true skip connections (default: False).
        encdec_compress (int): Compression factor (default: 2).
        encdec_lstm (int): Number of LSTM layers (default: 2).
        decoder_trim_right_ratio (float): Ratio for trimming the decoder output (default: 1.0).
        decoder_final_activation (Optional[str]): Final activation function (default: None).
        decoder_final_activation_params (Optional[dict]): Parameters for final activation
            (default: None).
        quantizer_n_q (int): Number of quantization codes (default: 8).
        quantizer_bins (int): Number of bins for quantization (default: 1024).
        quantizer_decay (float): Decay factor for quantization (default: 0.99).
        quantizer_kmeans_init (bool): Whether to initialize with k-means (default: True).
        quantizer_kmeans_iters (int): Number of iterations for k-means (default: 50).
        quantizer_threshold_ema_dead_code (int): Threshold for dead code (default: 2).
        quantizer_target_bandwidth (List[float]): Target bandwidths for quantization
            (default: [7.5, 15]).

    Returns:
        None: This constructor does not return any value.

    Examples:
        generator = SoundStreamGenerator(sample_rate=24000)
        output_audio, commit_loss, quantization_loss, resyn_audio_real = generator(
            input_tensor
        )

    Note:
        The input tensor for the forward method should have a shape of (B, 1, T),
        where B is the batch size and T is the length of the audio sequence.
    """

    @typechecked
    def __init__(
        self,
        sample_rate: int = 24000,
        hidden_dim: int = 128,
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
        """Initialize SoundStream Generator.

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
            >>> audio_input = torch.randn(8, 24000)  # Example audio input
            >>> model = SoundStream()
            >>> output = model.forward(audio_input, forward_generator=True)
            >>> print(output['loss'].item())  # Accessing the loss value

        Note:
            This method performs a forward pass through the generator or
            discriminator based on the `forward_generator` flag. If
            `forward_generator` is set to True, it processes the audio input
            through the generator; otherwise, it forwards through the
            discriminator.
        """
        encoder_out = self.encoder(x)
        max_idx = len(self.target_bandwidths) - 1

        # randomly pick up one bandwidth
        bw = self.target_bandwidths[random.randint(0, max_idx)]

        # Forward quantizer
        quantized, _, _, commit_loss = self.quantizer(encoder_out, self.frame_rate, bw)

        quantization_loss = self.l1_quantization_loss(
            encoder_out, quantized.detach()
        ) + self.l2_quantization_loss(encoder_out, quantized.detach())

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
            >>> model = SoundStream(...)
            >>> input_audio = torch.randn(1, 24000)  # Simulated audio input
            >>> codes = model.encode(input_audio)
            >>> print(codes.shape)  # Output shape should be (T_code, N_stream)

        Note:
            The input tensor should have a shape of (B, T_wav) where B is the batch
            size and T_wav is the length of the audio waveform. The output will be
            the encoded representation of the audio in terms of codes.
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
        Run decoding.

        This method takes neural codes as input and generates a waveform.

        Args:
            x (Tensor): Input codes (T_code, N_stream). The shape of the input
            tensor should correspond to the encoded representations of the audio
            signals.

        Returns:
            Tensor: Generated waveform (T_wav,). This tensor contains the
            reconstructed audio signal derived from the input codes.

        Examples:
            >>> model = SoundStream()
            >>> codes = torch.randn(100, 8)  # Example shape for codes
            >>> waveform = model.decode(codes)
            >>> print(waveform.shape)  # Output shape will be (T_wav,)
        """
        quantized = self.quantizer.decode(codes)
        resyn_audio = self.decoder(quantized)
        return resyn_audio


class SoundStreamDiscriminator(nn.Module):
    """
    SoundStream discriminator module.

    This module implements a multi-scale and complex STFT discriminator for the
    SoundStream model. It is designed to distinguish between real and generated
    audio signals, using multiple scales of feature extraction and
    complex short-time Fourier transform.

    Attributes:
        msd (HiFiGANMultiScaleDiscriminator): Multi-scale discriminator component.
        complex_stft_d (ComplexSTFTDiscriminator): Complex STFT discriminator component.

    Args:
        scales (int): Number of multi-scales for the discriminator.
        scale_downsample_pooling (str): Pooling module name for downsampling of the
            inputs.
        scale_downsample_pooling_params (Dict[str, Any]): Parameters for the above
            pooling module.
        scale_discriminator_params (Dict[str, Any]): Parameters for HiFi-GAN scale
            discriminator module.
        scale_follow_official_norm (bool): Whether to follow the norm setting of the
            official implementation. The first discriminator uses spectral norm
            and the other discriminators use weight norm.
        complexstft_discriminator_params (Dict[str, Any]): Parameters for the
            complex STFT discriminator module.

    Examples:
        >>> discriminator = SoundStreamDiscriminator(scales=3)
        >>> input_tensor = torch.randn(8, 1, 16000)  # Batch of 8 audio signals
        >>> outputs = discriminator(input_tensor)
        >>> len(outputs)  # Outputs will be a list containing outputs from both
        ...              # multi-scale and complex STFT discriminators.
    """

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
        """Initialize SoundStream Discriminator module.

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
        self.complex_stft_d = ComplexSTFTDiscriminator(
            **complexstft_discriminator_params
        )

    def forward(self, x: torch.Tensor) -> List[List[torch.Tensor]]:
        """
        Perform forward propagation for the SoundStream model.

        This method either runs the generator or discriminator depending on the
        value of the `forward_generator` flag. If `forward_generator` is set to
        True, the method will compute the generator's output and loss; if set
        to False, it will compute the discriminator's output and loss.

        Args:
            audio (torch.Tensor): Audio waveform tensor of shape (B, T_wav),
                where B is the batch size and T_wav is the number of audio
                samples.
            forward_generator (bool): A flag indicating whether to forward the
                generator (True) or the discriminator (False). Defaults to True.

        Returns:
            Dict[str, Any]:
                - loss (Tensor): A scalar tensor representing the total loss.
                - stats (Dict[str, float]): A dictionary containing various
                  statistics to be monitored during training.
                - weight (Tensor): A tensor summarizing the weights for loss
                  computation.
                - optim_idx (int): An integer indicating the optimizer index
                  (0 for generator and 1 for discriminator).

        Examples:
            >>> audio_input = torch.randn(4, 16000)  # Example audio input
            >>> model = SoundStream()  # Initialize the model
            >>> output = model.forward(audio_input, forward_generator=True)
            >>> print(output['loss'])  # Access the computed loss

        Note:
            Ensure that the audio tensor is properly shaped and normalized
            before passing it to the forward method.

        Raises:
            ValueError: If the audio tensor is not of shape (B, T_wav).
        """
        msd_outs = self.msd(x)
        complex_stft_outs = self.complex_stft_d(x)
        return msd_outs + complex_stft_outs

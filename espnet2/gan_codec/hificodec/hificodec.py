import functools
import math
import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typeguard import typechecked

from espnet2.gan_codec.abs_gan_codec import AbsGANCodec
from espnet2.gan_codec.hificodec.module import (
    Encoder,
    Generator,
    GroupResidualVectorQuantization,
)
from espnet2.gan_codec.shared.discriminator.msstft_discriminator import (
    MultiScaleSTFTDiscriminator,
)
from espnet2.gan_codec.shared.loss.freq_loss import MultiScaleMelSpectrogramLoss
from espnet2.gan_codec.shared.loss.loss_balancer import Balancer
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


class HiFiCodec(AbsGANCodec):
    """
    HiFiCodec model for high-fidelity audio generation and encoding.

    This model implements a GAN-based codec for generating high-quality audio.
    It consists of a generator and discriminator, with options for various
    loss functions and features for improved audio fidelity.

    Attributes:
        generator (HiFiCodecGenerator): The generator module for audio synthesis.
        discriminator (HiFiCodecDiscriminator): The discriminator module for
            evaluating the quality of generated audio.
        generator_adv_loss (GeneratorAdversarialLoss): Adversarial loss for the
            generator.
        generator_reconstruct_loss (L1Loss): Reconstruction loss for generator.
        discriminator_adv_loss (DiscriminatorAdversarialLoss): Adversarial loss
            for the discriminator.
        use_feat_match_loss (bool): Flag to indicate whether to use feature
            matching loss.
        feat_match_loss (FeatureMatchLoss): Feature matching loss if enabled.
        use_mel_loss (bool): Flag to indicate whether to use mel spectrogram loss.
        mel_loss (MultiScaleMelSpectrogramLoss): Mel spectrogram loss if enabled.
        use_dual_decoder (bool): Flag to indicate whether to use a dual decoder.
        cache_generator_outputs (bool): Flag to indicate whether to cache generator
            outputs for efficiency.
        loss_balancer (Balancer): Loss balancer for adjusting the weights of
            different loss components.

    Args:
        sampling_rate (int): The sampling rate of the audio. Default is 16000.
        generator_params (Dict[str, Any]): Parameters for the generator module.
        discriminator_params (Dict[str, Any]): Parameters for the discriminator
            module.
        generator_adv_loss_params (Dict[str, Any]): Parameters for generator
            adversarial loss.
        discriminator_adv_loss_params (Dict[str, Any]): Parameters for
            discriminator adversarial loss.
        use_feat_match_loss (bool): Whether to use feature matching loss.
        feat_match_loss_params (Dict[str, Any]): Parameters for feature matching
            loss.
        use_mel_loss (bool): Whether to use mel loss.
        mel_loss_params (Dict[str, Any]): Parameters for mel loss.
        use_dual_decoder (bool): Whether to use dual decoder.
        lambda_quantization (float): Weight for quantization loss. Default is 1.0.
        lambda_reconstruct (float): Weight for reconstruction loss. Default is 1.0.
        lambda_commit (float): Weight for commitment loss. Default is 1.0.
        lambda_adv (float): Weight for adversarial loss. Default is 1.0.
        lambda_feat_match (float): Weight for feature matching loss. Default is 2.0.
        lambda_mel (float): Weight for mel loss. Default is 45.0.
        cache_generator_outputs (bool): Whether to cache generator outputs.
        use_loss_balancer (bool): Whether to use loss balancer.
        balance_ema_decay (float): EMA decay rate for loss balancer. Default is 0.99.

    Examples:
        >>> model = HiFiCodec()
        >>> audio_input = torch.randn(1, 16000)  # Simulated audio input
        >>> output = model.forward(audio_input)
        >>> print(output['loss'])  # Access the loss from output

    Note:
        The generator and discriminator parameters can be adjusted for
        specific use cases, such as changing the number of layers or kernel
        sizes.

    Raises:
        AssertionError: If dual decoder is enabled without mel loss.
    """

    @typechecked
    def __init__(
        self,
        sampling_rate: int = 16000,
        generator_params: Dict[str, Any] = {
            "hidden_dim": 256,
            "resblock_num": "1",
            "resblock_kernel_sizes": [3, 7, 11],
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "upsample_rates": [8, 5, 4, 2],
            "upsample_kernel_sizes": [16, 11, 8, 4],
            "upsample_initial_channel": 512,
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
                "in_channels": 1,
                "out_channels": 1,
                "filters": 32,
                "norm": "weight_norm",
                "n_ffts": [1024, 2048, 512, 256, 128],
                "hop_lengths": [256, 512, 128, 64, 32],
                "win_lengths": [1024, 2048, 512, 256, 128],
                "activation": "LeakyReLU",
                "activation_params": {"negative_slope": 0.2},
            },
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
                "bias": False,
                "downsample_scales": [2, 2, 4, 4, 1],
                "nonlinear_activation": "LeakyReLU",
                "nonlinear_activation_params": {"negative_slope": 0.1},
                "use_weight_norm": True,
                "use_spectral_norm": False,
            },
            "scale_follow_official_norm": False,
            "periods": [2, 3, 5, 7, 11],
            "periods_discriminator_params": {
                "in_channels": 1,
                "out_channels": 1,
                "kernel_sizes": [5, 3],
                "channels": 32,
                "downsample_scales": [3, 3, 3, 3, 1],
                "max_downsample_channels": 1024,
                "bias": False,
                "nonlinear_activation": "LeakyReLU",
                "nonlinear_activation_params": {"negative_slope": 0.1},
                "use_weight_norm": True,
                "use_spectral_norm": False,
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
            "fs": 16000,
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
        """Intialize HiFiCodec model."""
        super().__init__()

        # define modules
        generator_params.update(sample_rate=sampling_rate)
        self.generator = HiFiCodecGenerator(**generator_params)
        self.discriminator = HiFiCodecDiscriminator(**discriminator_params)
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
            lambda x, y: x * y, generator_params["upsample_rates"]
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
        Retrieve model meta-information.

        This method provides essential information about the HiFiCodec model's
        configuration, including the sampling rate, number of streams, frame
        shift, and code size per stream.

        Returns:
            Dict[str, Any]: A dictionary containing the following keys:
                - fs (int): The sampling rate of the model.
                - num_streams (int): The number of quantizer streams used in the
                  model.
                - frame_shift (int): The frame shift calculated based on the
                  upsample rates.
                - code_size_per_stream (List[int]): A list containing the code
                  size for each quantization stream.

        Examples:
            >>> model = HiFiCodec()
            >>> meta = model.meta_info()
            >>> print(meta)
            {'fs': 16000, 'num_streams': 8, 'frame_shift': 640,
             'code_size_per_stream': [1024, 1024, 1024, 1024, 1024, 1024,
             1024, 1024]}

        Note:
            This method is useful for understanding the configuration of the
            model and for debugging purposes.
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

        This method performs the forward pass of the HiFiCodec model. It can
        either execute the generator or discriminator forward pass based on the
        `forward_generator` flag. The output includes loss metrics and
        statistics that are useful for monitoring training progress.

        Args:
            audio (torch.Tensor):
                Audio waveform tensor with shape (B, T_wav), where B is the
                batch size and T_wav is the number of audio samples.
            forward_generator (bool):
                Flag to indicate whether to forward through the generator
                (True) or the discriminator (False).

        Returns:
            Dict[str, Any]:
                - loss (Tensor): Loss scalar tensor representing the total
                  computed loss.
                - stats (Dict[str, float]): Dictionary containing various
                  statistics to be monitored during training, such as
                  individual loss components.
                - weight (Tensor): Weight tensor used to summarize losses.
                - optim_idx (int): Index of the optimizer to be used (0 for
                  generator and 1 for discriminator).

        Examples:
            >>> model = HiFiCodec()
            >>> audio_input = torch.randn(2, 16000)  # Batch of 2 audio samples
            >>> output = model.forward(audio_input, forward_generator=True)
            >>> print(output['loss'], output['stats'])

        Note:
            Ensure that the input audio tensor is properly preprocessed
            and has the correct shape before calling this method.
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
            HiFiCodec model.

        This class implements the HiFiCodec architecture, which is a generative
        adversarial network (GAN) designed for high-fidelity audio generation.

        Attributes:
            generator (HiFiCodecGenerator): The generator module for audio synthesis.
            discriminator (HiFiCodecDiscriminator): The discriminator module for
                evaluating the generated audio.
            generator_adv_loss (GeneratorAdversarialLoss): Adversarial loss for the
                generator.
            generator_reconstruct_loss (torch.nn.L1Loss): Reconstruction loss for the
                generator.
            discriminator_adv_loss (DiscriminatorAdversarialLoss): Adversarial loss for
                the discriminator.
            use_feat_match_loss (bool): Flag to indicate whether to use feature
                matching loss.
            feat_match_loss (FeatureMatchLoss): Feature matching loss module.
            use_mel_loss (bool): Flag to indicate whether to use mel loss.
            mel_loss (MultiScaleMelSpectrogramLoss): Mel loss module.
            cache_generator_outputs (bool): Flag to cache generator outputs.
            loss_balancer (Optional[Balancer]): Loss balancer for managing multiple loss
                components.

        Args:
            sampling_rate (int): Sampling rate for audio (default: 16000).
            generator_params (Dict[str, Any]): Parameters for the generator module.
            discriminator_params (Dict[str, Any]): Parameters for the discriminator module.
            generator_adv_loss_params (Dict[str, Any]): Parameters for the generator
                adversarial loss.
            discriminator_adv_loss_params (Dict[str, Any]): Parameters for the
                discriminator adversarial loss.
            use_feat_match_loss (bool): Flag to use feature matching loss (default: True).
            feat_match_loss_params (Dict[str, Any]): Parameters for feature matching loss.
            use_mel_loss (bool): Flag to use mel loss (default: True).
            mel_loss_params (Dict[str, Any]): Parameters for mel loss.
            use_dual_decoder (bool): Flag to use dual decoder (default: True).
            lambda_quantization (float): Coefficient for quantization loss (default: 1.0).
            lambda_reconstruct (float): Coefficient for reconstruction loss (default: 1.0).
            lambda_commit (float): Coefficient for commitment loss (default: 1.0).
            lambda_adv (float): Coefficient for adversarial loss (default: 1.0).
            lambda_feat_match (float): Coefficient for feature matching loss (default: 2.0).
            lambda_mel (float): Coefficient for mel loss (default: 45.0).
            cache_generator_outputs (bool): Flag to cache generator outputs (default: False).
            use_loss_balancer (bool): Flag to use loss balancer (default: False).
            balance_ema_decay (float): Exponential moving average decay for balancing loss
                (default: 0.99).

        Examples:
            # Initialize the HiFiCodec model with default parameters
            hifi_codec = HiFiCodec()

            # Initialize the HiFiCodec model with custom parameters
            custom_hifi_codec = HiFiCodec(sampling_rate=22050,
                                           generator_params={"hidden_dim": 512})

        Note:
            The model expects audio input in the form of a PyTorch tensor of shape
            (B, T_wav) for training and inference.

        Todo:
            - Implement additional features for model customization.
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

        Args:
            x (Tensor): Input audio (T_wav,).

        Returns:
            Tensor: Generated codes (T_code, N_stream).

        Examples:
            >>> model = HiFiCodec()
            >>> audio_input = torch.randn(1, 16000)  # Simulated audio input
            >>> codes = model.encode(audio_input)
            >>> print(codes.shape)  # Expected output shape: (T_code, N_stream)

        Note:
            This method utilizes the generator's encode function to process the
            input audio and produce a set of neural codec representations.
        """
        # print(x.shape)
        return self.generator.encode(x)

    def decode(
        self,
        x: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Run decoding.

        This method takes encoded audio codes as input and generates a
        waveform tensor. The decoding process involves the reconstruction
        of the original audio from its encoded representation.

        Args:
            x (Tensor): Input codes (T_code, N_stream), where T_code is the
                length of the code sequence and N_stream is the number of
                streams in the codec.

        Returns:
            Tensor: Generated waveform (T_wav,), where T_wav is the length of
            the reconstructed audio waveform.

        Examples:
            >>> codec_input = torch.randn(100, 8)  # Example input codes
            >>> waveform = hi_fi_codec.decode(codec_input)
            >>> print(waveform.shape)
            torch.Size([T_wav,])  # Output shape will depend on the model
        """
        return self.generator.decode(x)


class HiFiCodecGenerator(nn.Module):
    """
    HiFiCodec generator module.

    This class implements the generator for the HiFiCodec model, which
    processes audio waveforms through encoding and decoding mechanisms.
    The generator uses a combination of an encoder, a quantizer, and a
    decoder to achieve high-fidelity audio synthesis.

    Attributes:
        encoder (Encoder): The encoder module that extracts features from
            the input audio.
        quantizer (GroupResidualVectorQuantization): The quantization module
            that compresses the encoded features.
        decoder (Generator): The decoder module that reconstructs audio from
            quantized features.
        target_bandwidths (List[float]): List of target bandwidths for
            quantization.
        sample_rate (int): The sample rate of the audio.
        frame_rate (int): The frame rate derived from the sample rate and
            upsample rates.

    Args:
        sample_rate (int): Sample rate of the input audio. Default is 16000.
        hidden_dim (int): Dimensionality of hidden layers. Default is 128.
        resblock_num (str): Number of residual blocks. Default is "1".
        resblock_kernel_sizes (List[int]): List of kernel sizes for residual
            blocks. Default is [3, 7, 11].
        resblock_dilation_sizes (List[List[int]]): List of dilation sizes for
            residual blocks. Default is [[1, 3, 5], [1, 3, 5], [1, 3, 5]].
        upsample_rates (List[int]): List of upsample rates. Default is [8, 5, 4, 2].
        upsample_kernel_sizes (List[int]): List of kernel sizes for upsampling.
            Default is [16, 11, 8, 4].
        upsample_initial_channel (int): Number of initial channels for the
            upsampling layer. Default is 512.
        quantizer_n_q (int): Number of quantizers. Default is 8.
        quantizer_bins (int): Number of quantization bins. Default is 1024.
        quantizer_decay (float): Decay rate for quantization. Default is 0.99.
        quantizer_kmeans_init (bool): Whether to initialize with k-means.
            Default is True.
        quantizer_kmeans_iters (int): Number of iterations for k-means. Default is 50.
        quantizer_threshold_ema_dead_code (int): Threshold for dead code.
            Default is 2.
        quantizer_target_bandwidth (List[float]): List of target bandwidths
            for quantization. Default is [7.5, 15].

    Examples:
        >>> generator = HiFiCodecGenerator()
        >>> input_audio = torch.randn(1, 1, 16000)  # (B, 1, T)
        >>> output = generator(input_audio)
        >>> print(output[0].shape)  # Resynthesized audio shape
        >>> print(output[1].shape)  # Commitment loss shape
        >>> print(output[2].shape)  # Quantization loss shape
        >>> print(output[3].shape)  # Resynthesized audio from encoder
    """

    @typechecked
    def __init__(
        self,
        sample_rate: int = 16000,
        hidden_dim: int = 128,
        resblock_num: str = "1",
        resblock_kernel_sizes: List[int] = [3, 7, 11],
        resblock_dilation_sizes: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_rates: List[int] = [8, 5, 4, 2],
        upsample_kernel_sizes: List[int] = [16, 11, 8, 4],
        upsample_initial_channel: int = 512,
        quantizer_n_q: int = 8,
        quantizer_bins: int = 1024,
        quantizer_decay: float = 0.99,
        quantizer_kmeans_init: bool = True,
        quantizer_kmeans_iters: int = 50,
        quantizer_threshold_ema_dead_code: int = 2,
        quantizer_target_bandwidth: List[float] = [7.5, 15],
    ):
        """Initialize HiFiCodec Generator.
        Args:
            TODO
        """
        super().__init__()

        # Initialize encoder
        self.encoder = Encoder(
            resblock_num=resblock_num,
            resblock_kernel_sizes=resblock_kernel_sizes,
            resblock_dilation_sizes=resblock_dilation_sizes,
            upsample_rates=upsample_rates,
            upsample_kernel_sizes=upsample_kernel_sizes,
        )

        # Initialize quantizer
        self.quantizer = GroupResidualVectorQuantization(
            hidden_dim=hidden_dim,
            quantizer_n_q=quantizer_n_q,
            quantizer_bins=quantizer_bins,
            quantizer_decay=quantizer_decay,
            quantizer_kmeans_init=quantizer_kmeans_init,
            quantizer_kmeans_iters=quantizer_kmeans_iters,
            quantizer_threshold_ema_dead_code=quantizer_threshold_ema_dead_code,
            quantizer_target_bandwidth=quantizer_target_bandwidth,
        )

        self.target_bandwidths = quantizer_target_bandwidth
        self.sample_rate = sample_rate
        self.frame_rate = math.ceil(sample_rate / np.prod(upsample_rates))

        self.decoder = Generator(
            upsample_rates=upsample_rates,
            upsample_kernel_sizes=upsample_kernel_sizes,
            upsample_initial_channel=upsample_initial_channel,
            resblock_num=resblock_num,
            resblock_kernel_sizes=resblock_kernel_sizes,
            resblock_dilation_sizes=resblock_dilation_sizes,
            out_dim=2 * hidden_dim,
        )

    def forward(self, x: torch.Tensor, use_dual_decoder: bool = False):
        """
        Perform generator forward.

        This method handles the forward pass for either the generator or
        discriminator, based on the `forward_generator` flag. It processes
        the input audio waveform tensor and computes the corresponding
        losses and statistics.

        Args:
            audio (Tensor): Audio waveform tensor of shape (B, T_wav).
            forward_generator (bool): If True, the forward pass is done
                through the generator; otherwise, it goes through the
                discriminator.

        Returns:
            Dict[str, Any]:
                - loss (Tensor): Loss scalar tensor computed during the
                  forward pass.
                - stats (Dict[str, float]): Statistics for monitoring,
                  including individual loss components.
                - weight (Tensor): Weight tensor summarizing losses.
                - optim_idx (int): Optimizer index (0 for G and 1 for D).

        Examples:
            >>> model = HiFiCodec()
            >>> audio_input = torch.randn(8, 16000)  # Batch of 8, 1 second audio
            >>> output = model.forward(audio_input, forward_generator=True)
            >>> print(output['loss'].item())  # Access the computed loss

        Note:
            The method will call either `_forward_generator` or
            `_forward_discrminator` based on the value of
            `forward_generator`.
        """
        encoder_out = self.encoder(x)
        # print("x shape:", x.shape)
        # print("encoder_out shape:", encoder_out.shape)
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
            >>> import torch
            >>> model = HiFiCodec()
            >>> input_audio = torch.randn(1, 16000)  # Example audio tensor
            >>> encoded_codes = model.encode(input_audio)
            >>> print(encoded_codes.shape)  # Output shape: (T_code, N_stream)

        Note:
            The input tensor should have the shape (B, T_wav) where B is the batch
            size and T_wav is the number of audio samples.
        """
        if x.dim() == 1:
            x = x.view(1, 1, -1)
        elif x.dim() == 2:
            x = x.unsqueeze(1)

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

        This method takes input codes generated by the encoder and produces a
        waveform. It is an essential step in the HiFiCodec pipeline, converting
        compressed representations back into audio signals.

        Args:
            x (Tensor): Input codes (T_code, N_stream). These are the codes
                produced by the encoder, which represent the compressed audio
                data.

        Returns:
            Tensor: Generated waveform (T_wav,). This is the output audio signal
            reconstructed from the input codes.

        Examples:
            >>> codec = HiFiCodec()
            >>> input_codes = torch.randn(10, 8)  # Example input codes
            >>> waveform = codec.decode(input_codes)
            >>> print(waveform.shape)  # Output shape will depend on the model

        Note:
            The input codes should be of the shape (T_code, N_stream) where
            T_code is the length of the code sequence and N_stream is the number
            of streams used in the codec.
        """
        quantized = self.quantizer.decode(codes)
        resyn_audio = self.decoder(quantized)

        return resyn_audio


class HiFiCodecDiscriminator(nn.Module):
    """
    HiFiCodec discriminator module.

    This class implements a discriminator for the HiFiCodec model, which uses
    multi-scale and multi-period discriminators to evaluate the quality of
    generated audio signals.

    Attributes:
        msstft (MultiScaleSTFTDiscriminator): Multi-scale STFT discriminator.
        msd (HiFiGANMultiScaleDiscriminator): Multi-scale discriminator.
        mpd (HiFiGANMultiPeriodDiscriminator): Multi-period discriminator.

    Args:
        msstft_discriminator_params (Dict[str, Any]): Parameters for multi-scales
            STFT discriminator module.
        scales (int): Number of multi-scales.
        scale_downsample_pooling (str): Pooling module name for downsampling of
            the inputs.
        scale_downsample_pooling_params (Dict[str, Any]): Parameters for the
            above pooling module.
        scale_discriminator_params (Dict[str, Any]): Parameters for HiFi-GAN
            scale discriminator module.
        scale_follow_official_norm (bool): Flag to follow the official
            normalization.
        periods (List[int]): List of periods for multi-period discriminator.
        periods_discriminator_params (Dict[str, Any]): Parameters for HiFi-GAN
            period discriminator module. The period parameter will be overwritten.

    Examples:
        >>> discriminator = HiFiCodecDiscriminator()
        >>> input_tensor = torch.randn(8, 1, 16000)  # Batch of 8 audio signals
        >>> outputs = discriminator(input_tensor)
        >>> len(outputs)  # Check the number of outputs from the discriminators
        8
    """

    def __init__(
        self,
        # Multi-scale STFT discriminator related
        msstft_discriminator_params: Dict[str, Any] = {
            "in_channels": 1,
            "out_channels": 1,
            "filters": 32,
            "norm": "weight_norm",
            "n_ffts": [1024, 2048, 512, 256, 128],
            "hop_lengths": [256, 512, 128, 64, 32],
            "win_lengths": [1024, 2048, 512, 256, 128],
            "activation": "LeakyReLU",
            "activation_params": {"negative_slope": 0.2},
        },
        # Multi-scale discriminator related
        scales: int = 3,
        scale_downsample_pooling: str = "AvgPool1d",
        # set bias to False
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
            "bias": False,
            "downsample_scales": [2, 2, 4, 4, 1],
            "nonlinear_activation": "LeakyReLU",
            "nonlinear_activation_params": {"negative_slope": 0.1},
            "use_weight_norm": True,
            "use_spectral_norm": False,
        },
        scale_follow_official_norm: bool = False,
        # Multi-period discriminator related
        periods: List[int] = [2, 3, 5, 7, 11],
        # set bias to False
        periods_discriminator_params: Dict[str, Any] = {
            "in_channels": 1,
            "out_channels": 1,
            "kernel_sizes": [5, 3],
            "channels": 32,
            "downsample_scales": [3, 3, 3, 3, 1],
            "max_downsample_channels": 1024,
            "bias": False,
            "nonlinear_activation": "LeakyReLU",
            "nonlinear_activation_params": {"negative_slope": 0.1},
            "use_weight_norm": True,
            "use_spectral_norm": False,
        },
    ):
        """Initialize HiFiCodec Discriminator module.

        Args:
            msstft_discriminator_params (Dict[str, Any]): Parameters for multi-scales
                STFT discriminator module.
            scales (int): Number of multi-scales.
            sclae_downsample_pooling (str): Pooling module name for downsampling of the
                inputs.
            scale_downsample_pooling_params (Dict[str, Any]): Parameters for the above
                pooling module.
            scale_discriminator_params (Dict[str, Any]): Parameters for hifi-gan scale
                discriminator module.
            periods (List[int]): List of periods.
            discriminator_params (Dict[str, Any]): Parameters for hifi-gan period
                discriminator module. The period parameter will be overwritten.
        """
        super().__init__()

        self.msstft = MultiScaleSTFTDiscriminator(
            **msstft_discriminator_params,
        )
        self.msd = HiFiGANMultiScaleDiscriminator(
            scales=scales,
            downsample_pooling=scale_downsample_pooling,
            downsample_pooling_params=scale_downsample_pooling_params,
            discriminator_params=scale_discriminator_params,
            follow_official_norm=scale_follow_official_norm,
        )
        self.mpd = HiFiGANMultiPeriodDiscriminator(
            periods=periods,
            discriminator_params=periods_discriminator_params,
        )

    def forward(self, x: torch.Tensor) -> List[List[torch.Tensor]]:
        """
            Perform forward propagation for the HiFiCodec model.

        This method decides whether to forward the audio through the generator
        or the discriminator based on the `forward_generator` flag. It computes
        the loss and statistics based on the selected path.

        Args:
            audio (Tensor): Audio waveform tensor of shape (B, T_wav), where B is
                the batch size and T_wav is the length of the audio waveform.
            forward_generator (bool): If True, forwards the audio through the
                generator; if False, forwards it through the discriminator.

        Returns:
            Dict[str, Any]:
                - loss (Tensor): Loss scalar tensor.
                - stats (Dict[str, float]): Statistics to be monitored during training.
                - weight (Tensor): Weight tensor to summarize losses.
                - optim_idx (int): Index for the optimizer (0 for generator and
                  1 for discriminator).

        Examples:
            >>> model = HiFiCodec()
            >>> audio_input = torch.randn(8, 16000)  # Batch of 8 audio samples
            >>> output = model.forward(audio_input, forward_generator=True)
            >>> print(output['loss'], output['stats'])

        Note:
            Ensure that the audio input tensor is properly shaped and contains
            valid waveform data. The `kwargs` can be used to pass additional
            parameters needed for either generator or discriminator forward pass.
        """
        # 5 scale list of [fmap + [logit]]
        msstft_outs = self.msstft(x)
        # 3 scale 4 of each layer
        msd_outs = self.msd(x)
        # 5 period 4 of each layer
        mpd_outs = self.mpd(x)

        return msstft_outs + msd_outs + mpd_outs

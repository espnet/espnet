# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from typing import Any, Dict, List

import torch

from espnet2.gan_codec.shared.discriminator.msstft_discriminator import (
    MultiScaleSTFTDiscriminator,
)
from espnet2.gan_codec.soundstream.soundstream import SoundStream


class Encodec(SoundStream):
    """
    Encodec Model for audio encoding and decoding.

    This model is based on the SoundStream architecture with modifications to
    the discriminator and loss balancer. It is designed for efficient audio
    encoding and reconstruction tasks.

    For more details, refer to the paper:
    https://arxiv.org/abs/2210.13438

    Attributes:
        discriminator (EncodecDiscriminator): The discriminator component of
            the model.

    Args:
        sampling_rate (int): The sampling rate of the audio. Default is 24000.
        generator_params (Dict[str, Any]): Parameters for the generator model.
        discriminator_params (Dict[str, Any]): Parameters for the discriminator
            model.
        generator_adv_loss_params (Dict[str, Any]): Parameters for the generator
            adversarial loss.
        discriminator_adv_loss_params (Dict[str, Any]): Parameters for the
            discriminator adversarial loss.
        use_feat_match_loss (bool): Whether to use feature matching loss.
            Default is True.
        feat_match_loss_params (Dict[str, Any]): Parameters for feature matching
            loss.
        use_mel_loss (bool): Whether to use mel loss. Default is True.
        mel_loss_params (Dict[str, Any]): Parameters for mel loss.
        use_dual_decoder (bool): Whether to use dual decoding mechanism.
            Default is True.
        lambda_quantization (float): Weight for quantization loss. Default is
            1.0.
        lambda_reconstruct (float): Weight for reconstruction loss. Default is
            1.0.
        lambda_commit (float): Weight for commitment loss. Default is 1.0.
        lambda_adv (float): Weight for adversarial loss. Default is 1.0.
        lambda_feat_match (float): Weight for feature matching loss. Default is
            2.0.
        lambda_mel (float): Weight for mel loss. Default is 45.0.
        cache_generator_outputs (bool): Whether to cache generator outputs.
            Default is False.
        use_loss_balancer (bool): Whether to use loss balancing. Default is
            False.
        balance_ema_decay (float): Exponential moving average decay for loss
            balancing. Default is 0.99.

    Examples:
        # Creating an instance of the Encodec model
        model = Encodec(sampling_rate=24000, use_feat_match_loss=True)
    """

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
        # Note(Jinchuan): re-apply everything except the discriminator config.
        # Init discriminator from default config and then override it.
        super(Encodec, self).__init__(
            sampling_rate=sampling_rate,
            generator_params=generator_params,
            generator_adv_loss_params=generator_adv_loss_params,
            discriminator_adv_loss_params=discriminator_adv_loss_params,
            use_feat_match_loss=use_feat_match_loss,
            feat_match_loss_params=feat_match_loss_params,
            use_mel_loss=use_mel_loss,
            mel_loss_params=mel_loss_params,
            use_dual_decoder=use_dual_decoder,
            lambda_quantization=lambda_quantization,
            lambda_reconstruct=lambda_reconstruct,
            lambda_commit=lambda_commit,
            lambda_adv=lambda_adv,
            lambda_feat_match=lambda_feat_match,
            lambda_mel=lambda_mel,
            cache_generator_outputs=cache_generator_outputs,
            use_loss_balancer=use_loss_balancer,
            balance_ema_decay=balance_ema_decay,
        )

        self.discriminator = EncodecDiscriminator(**discriminator_params)


class EncodecDiscriminator(torch.nn.Module):
    """
        Encodec Discriminator with only Multi-Scale STFT discriminator module.

    This class implements the Encodec Discriminator, which utilizes a
    Multi-Scale Short-Time Fourier Transform (STFT) for analyzing the
    input signals. It is designed to work in conjunction with the
    Encodec model for adversarial training.

    Attributes:
        msstft (MultiScaleSTFTDiscriminator): The Multi-Scale STFT
            discriminator module.

    Args:
        msstft_discriminator_params (Dict[str, Any]): A dictionary of
            parameters for initializing the Multi-Scale STFT discriminator
            with the following keys:
            - in_channels (int): Number of input channels.
            - out_channels (int): Number of output channels.
            - filters (int): Number of filters in convolutions.
            - norm (str): Normalization choice of Convolutional layers.
            - n_ffts (Sequence[int]): Size of FFT for each scale.
            - hop_lengths (Sequence[int]): Length of hop between STFT windows
                for each scale.
            - win_lengths (Sequence[int]): Window size for each scale.
            - activation (str): Activation function choice of convolutional
                layer.
            - activation_params (Dict[str, Any]): Parameters for the
                activation function.

    Examples:
        >>> discriminator = EncodecDiscriminator()
        >>> input_tensor = torch.randn(8, 1, 1024)  # Batch size of 8, 1 channel, 1024 samples
        >>> outputs = discriminator(input_tensor)
        >>> print(len(outputs))  # Number of scales
        >>> print(len(outputs[0]))  # Number of outputs for the first scale

    Returns:
        List[List[Tensor]]: A list of lists of each discriminator output,
        which consists of each layer output tensors. Only one discriminator
        is used here, but the output is structured as a list of lists for
        consistency.

    Raises:
        ValueError: If any of the parameters are invalid during initialization.
    """

    def __init__(
        self,
        msstft_discriminator_params: Dict[str, Any] = {
            "in_channels": 1,
            "out_channels": 1,
            "filters": 32,
            "norm": "weight_norm",
            "n_fft": [1024, 2048, 512, 256, 128],
            "hop_lengths": [256, 512, 128, 64, 32],
            "win_lengths": [1024, 2048, 512, 256, 128],
            "activation": "LeakyReLU",
            "activation_params": {"negative_slope: 0.3"},
        },
    ):
        """Initialize Encodec Discriminator module.

        Args: msstft_discriminator_params (Dict[str, Any]) with following arguments:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            filters (int): Number of filters in convolutions.
            norm (str): normalization choice of Convolutional layers
            n_ffts (Sequence[int]): Size of FFT for each scale.
            hop_lengths (Sequence[int]): Length of hop between STFT windows for
                each scale.
            win_lengths (Sequence[int]): Window size for each scale.
            activation (str): activation function choice of convolutional layer
            activation_params (Dict[str, Any]): parameters for activation function)
        """

        super().__init__()

        self.msstft = MultiScaleSTFTDiscriminator(**msstft_discriminator_params)

    def forward(self, x: torch.Tensor) -> List[List[torch.Tensor]]:
        """
            Encodec Discriminator with only Multi-Scale STFT discriminator module.

        This class implements the Encodec Discriminator, which utilizes a
        Multi-Scale Short-Time Fourier Transform (STFT) for evaluating
        the quality of generated audio signals. The discriminator aims
        to distinguish between real and generated audio, contributing to
        the adversarial training process.

        Attributes:
            msstft (MultiScaleSTFTDiscriminator): The multi-scale STFT
                discriminator instance used for feature extraction.

        Args:
            msstft_discriminator_params (Dict[str, Any]): Parameters for the
                Multi-Scale STFT Discriminator, including:
                - in_channels (int): Number of input channels.
                - out_channels (int): Number of output channels.
                - filters (int): Number of filters in convolutions.
                - norm (str): Normalization choice for convolutional layers.
                - n_ffts (Sequence[int]): Sizes of FFT for each scale.
                - hop_lengths (Sequence[int]): Length of hop between STFT
                    windows for each scale.
                - win_lengths (Sequence[int]): Window sizes for each scale.
                - activation (str): Activation function choice for
                    convolutional layers.
                - activation_params (Dict[str, Any]): Parameters for the
                    activation function.

        Examples:
            >>> discriminator = EncodecDiscriminator()
            >>> input_signal = torch.randn(8, 1, 16000)  # Batch of 8, 1 channel, 16000 samples
            >>> outputs = discriminator(input_signal)
            >>> print(len(outputs))  # Number of scales
            >>> print(len(outputs[0]))  # Number of layers in the first scale
        """

        msstft_out = self.msstft(x)

        return msstft_out

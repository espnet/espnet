# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from typing import Any, Dict, List

import torch

from espnet2.gan_codec.shared.discriminator.msstft_discriminator import (
    MultiScaleSTFTDiscriminator,
)
from espnet2.gan_codec.soundstream.soundstream import SoundStream


class Encodec(SoundStream):
    """Encodec Model: https://arxiv.org/abs/2210.13438

    The key differences between this and SoundStream are the discriminator
    and loss balancer. Check SoundStream model for many details
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
    """Encodec Discriminator with only Multi-Scale STFT discriminator module"""

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
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).

        Returns:
            List[List[Tensor]]: List of list of each discriminator outputs,
                which consists of each layer output tensors. Only one
                discriminator here, but still make it as List of List for
                consistency.
        """

        msstft_out = self.msstft(x)

        return msstft_out

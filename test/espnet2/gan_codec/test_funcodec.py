# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Test code for SoundStream modules."""

import numpy as np  # noqa
import pytest
import torch

from espnet2.gan_codec.funcodec.funcodec import FunCodecDiscriminator, FunCodecGenerator
from espnet2.gan_codec.shared.loss.freq_loss import MultiScaleMelSpectrogramLoss
from espnet2.gan_tts.hifigan.loss import (
    DiscriminatorAdversarialLoss,
    FeatureMatchLoss,
    GeneratorAdversarialLoss,
)


def make_generator_args(**kwargs):
    default = dict(
        sample_rate=120,
        hidden_dim=2,
        codebook_dim=2,
        encdec_channels=3,
        encdec_n_filters=32,
        encdec_n_residual_layers=1,
        encdec_ratios=[(4, 1), (2, 1)],
        encdec_activation="ELU",
        encdec_activation_params={"alpha": 1.0},
        encdec_norm="weight_norm",
        encdec_norm_params={},
        encdec_kernel_size=1,
        encdec_residual_kernel_size=1,
        encdec_last_kernel_size=1,
        encdec_dilation_base=2,
        encdec_causal=False,
        encdec_pad_mode="reflect",
        encdec_true_skip=False,
        encdec_compress=2,
        encdec_lstm=1,
        decoder_trim_right_ratio=1.0,
        decoder_final_activation=None,
        decoder_final_activation_params=None,
        quantizer_n_q=2,
        quantizer_bins=4,
        quantizer_decay=0.99,
        quantizer_kmeans_init=True,
        quantizer_kmeans_iters=1,
        quantizer_threshold_ema_dead_code=2,
        quantizer_target_bandwidth=[7.5, 15],
        codec_domain=["mag_phase", "mag_phase"],
        domain_conf={"n_fft": 16, "hop_length": 4},
    )
    default.update(kwargs)
    return default


def make_discriminator_args(**kwargs):
    defaults = dict(
        scales=2,
        scale_downsample_pooling="AvgPool1d",
        scale_downsample_pooling_params={
            "kernel_size": 4,
            "stride": 2,
            "padding": 2,
        },
        scale_discriminator_params={
            "in_channels": 1,
            "out_channels": 1,
            "kernel_sizes": [15, 41, 5, 3],
            "channels": 16,
            "max_downsample_channels": 16,
            "max_groups": 16,
            "bias": True,
            "downsample_scales": [2, 2],
            "nonlinear_activation": "LeakyReLU",
            "nonlinear_activation_params": {"negative_slope": 0.1},
        },
        scale_follow_official_norm=False,
        # ComplexSTFT discriminator related
        complexstft_discriminator_params={
            "in_channels": 1,
            "channels": 4,
            "strides": [[1, 2], [2, 2], [1, 2], [2, 2], [1, 2], [2, 2]],
            "chan_mults": [1, 2, 4, 4, 8, 8],
            "n_fft": 16,
            "hop_length": 4,
            "win_length": 16,
            "stft_normalized": False,
        },
        periods=[2, 3],
        period_discriminator_params={
            "in_channels": 1,
            "out_channels": 1,
            "kernel_sizes": [5, 3],
            "channels": 8,
            "downsample_scales": [3, 3],
            "max_downsample_channels": 32,
            "bias": True,
            "nonlinear_activation": "LeakyReLU",
            "nonlinear_activation_params": {"negative_slope": 0.1},
            "use_weight_norm": True,
            "use_spectral_norm": False,
        },
    )
    defaults.update(kwargs)
    return defaults


def make_mel_loss_args(**kwargs):
    defaults = dict(
        fs=120,
        range_start=3,
        range_end=4,
        window="hann",
        n_mels=2,
        fmin=None,
        fmax=None,
        log_base=10.0,
    )
    defaults.update(kwargs)
    return defaults


@pytest.mark.skipif(
    "1.6" in torch.__version__,
    reason="group conv in pytorch 1.6 has an issue. "
    "See https://github.com/pytorch/pytorch/issues/42446.",
)
@pytest.mark.parametrize(
    "dict_g, dict_d, dict_loss, average, include",
    [
        ({}, {}, {}, True, True),
        ({}, {}, {}, False, False),
        ({"quantizer_kmeans_init": False}, {}, {}, False, True),
        ({"encdec_true_skip": True}, {}, {}, True, True),
    ],
)
def test_funcodec(dict_g, dict_d, dict_loss, average, include):
    batch_size = 2
    batch_length = 128
    args_g = make_generator_args(**dict_g)
    args_d = make_discriminator_args(**dict_d)
    args_loss = make_mel_loss_args(**dict_loss)
    y = torch.randn(batch_size, 1, batch_length)
    model_g = FunCodecGenerator(**args_g)
    model_d = FunCodecDiscriminator(**args_d)
    aux_criterion = MultiScaleMelSpectrogramLoss(**args_loss)
    feat_match_criterion = FeatureMatchLoss(
        average_by_layers=average,
        average_by_discriminators=average,
        include_final_outputs=include,
    )
    gen_adv_criterion = GeneratorAdversarialLoss(
        average_by_discriminators=average,
    )
    dis_adv_criterion = DiscriminatorAdversarialLoss(
        average_by_discriminators=average,
    )
    optimizer_g = torch.optim.AdamW(model_g.parameters())
    optimizer_d = torch.optim.AdamW(model_d.parameters())
    # check generator trainable
    y_hat, _, _, _ = model_g(y)
    p_hat = model_d(y_hat)
    aux_loss = aux_criterion(y_hat, y)
    adv_loss = gen_adv_criterion(p_hat)
    with torch.no_grad():
        p = model_d(y)
    fm_loss = feat_match_criterion(p_hat, p)
    loss_g = adv_loss + aux_loss + fm_loss
    optimizer_g.zero_grad()
    loss_g.backward()
    optimizer_g.step()

    # check discriminator trainable
    p = model_d(y)
    p_hat = model_d(y_hat.detach())
    real_loss, fake_loss = dis_adv_criterion(p_hat, p)
    loss_d = real_loss + fake_loss
    optimizer_d.zero_grad()
    loss_d.backward()
    optimizer_d.step()

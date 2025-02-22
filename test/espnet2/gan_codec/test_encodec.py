# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Test code for SoundStream modules."""

import numpy as np  # noqa
import pytest
import torch

from espnet2.gan_codec.encodec.encodec import EncodecDiscriminator
from espnet2.gan_codec.shared.loss.freq_loss import MultiScaleMelSpectrogramLoss
from espnet2.gan_codec.soundstream.soundstream import SoundStreamGenerator
from espnet2.gan_tts.hifigan.loss import (
    DiscriminatorAdversarialLoss,
    FeatureMatchLoss,
    GeneratorAdversarialLoss,
)


def make_generator_args(**kwargs):
    default = dict(
        sample_rate=120,
        hidden_dim=2,
        encdec_channels=1,
        encdec_n_filters=32,
        encdec_n_residual_layers=1,
        encdec_ratios=[2, 2],
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
    )
    default.update(kwargs)
    return default


def make_discriminator_args(**kwargs):
    defaults = {
        "msstft_discriminator_params": {
            "filters": 32,
            "in_channels": 1,
            "out_channels": 1,
            "sep_channels": False,
            "norm": "weight_norm",
            "n_ffts": [32, 16],
            "hop_lengths": [8, 4],
            "win_lengths": [32, 16],
            "activation": "LeakyReLU",
            "activation_params": {"negative_slope": 0.3},
        }
    }
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
def test_encodec(dict_g, dict_d, dict_loss, average, include):
    batch_size = 2
    batch_length = 128
    args_g = make_generator_args(**dict_g)
    args_d = make_discriminator_args(**dict_d)
    args_loss = make_mel_loss_args(**dict_loss)
    y = torch.randn(batch_size, 1, batch_length)
    model_g = SoundStreamGenerator(**args_g)
    model_d = EncodecDiscriminator(**args_d)
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

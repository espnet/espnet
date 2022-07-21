# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Test code for HiFi-GAN modules."""

import numpy as np
import pytest
import torch

from espnet2.gan_tts.hifigan import (
    HiFiGANGenerator,
    HiFiGANMultiScaleMultiPeriodDiscriminator,
)
from espnet2.gan_tts.hifigan.loss import (
    DiscriminatorAdversarialLoss,
    FeatureMatchLoss,
    GeneratorAdversarialLoss,
    MelSpectrogramLoss,
)


def make_hifigan_generator_args(**kwargs):
    defaults = dict(
        in_channels=5,
        out_channels=1,
        channels=32,
        kernel_size=7,
        upsample_scales=(2, 2),
        upsample_kernel_sizes=(4, 4),
        resblock_kernel_sizes=(3, 7),
        resblock_dilations=[(1, 3), (1, 3)],
        use_additional_convs=True,
        bias=True,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.1},
        use_weight_norm=True,
    )
    defaults.update(kwargs)
    return defaults


def make_hifigan_multi_scale_multi_period_discriminator_args(**kwargs):
    defaults = dict(
        scales=2,
        scale_downsample_pooling="AvgPool1d",
        scale_downsample_pooling_params={"kernel_size": 4, "stride": 2, "padding": 2,},
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
        follow_official_norm=False,
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
        n_fft=16,
        hop_length=4,
        win_length=None,
        window="hann",
        n_mels=2,
        fmin=None,
        fmax=None,
        center=True,
        normalized=False,
        onesided=True,
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
        ({}, {"scales": 1}, {}, False, True),
        ({}, {"periods": [2]}, {}, False, True),
        ({}, {"scales": 1, "periods": [2]}, {}, False, True),
        ({}, {"follow_official_norm": True}, {}, False, True),
        ({"use_additional_convs": False}, {}, {}, False, True),
        ({"global_channels": 4}, {}, {}, True, True),
    ],
)
def test_hifigan_generator_and_discriminator_and_loss(
    dict_g, dict_d, dict_loss, average, include
):
    batch_size = 2
    batch_length = 128
    args_g = make_hifigan_generator_args(**dict_g)
    args_d = make_hifigan_multi_scale_multi_period_discriminator_args(**dict_d)
    args_loss = make_mel_loss_args(**dict_loss)
    y = torch.randn(batch_size, 1, batch_length)
    c = torch.randn(
        batch_size,
        args_g["in_channels"],
        batch_length // np.prod(args_g["upsample_scales"]),
    )
    g = None
    if args_g.get("global_channels") is not None:
        g = torch.randn(batch_size, args_g["global_channels"], 1)
    model_g = HiFiGANGenerator(**args_g)
    model_d = HiFiGANMultiScaleMultiPeriodDiscriminator(**args_d)
    aux_criterion = MelSpectrogramLoss(**args_loss)
    feat_match_criterion = FeatureMatchLoss(
        average_by_layers=average,
        average_by_discriminators=average,
        include_final_outputs=include,
    )
    gen_adv_criterion = GeneratorAdversarialLoss(average_by_discriminators=average,)
    dis_adv_criterion = DiscriminatorAdversarialLoss(average_by_discriminators=average,)
    optimizer_g = torch.optim.AdamW(model_g.parameters())
    optimizer_d = torch.optim.AdamW(model_d.parameters())

    # check generator trainable
    y_hat = model_g(c, g=g)
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


try:
    import parallel_wavegan  # NOQA

    is_parallel_wavegan_available = True
except ImportError:
    is_parallel_wavegan_available = False


@pytest.mark.skipif(
    not is_parallel_wavegan_available, reason="parallel_wavegan is not installed."
)
def test_parallel_wavegan_compatibility():
    from parallel_wavegan.models import HiFiGANGenerator as PWGHiFiGANGenerator

    model_pwg = PWGHiFiGANGenerator(**make_hifigan_generator_args())
    model_espnet2 = HiFiGANGenerator(**make_hifigan_generator_args())
    model_espnet2.load_state_dict(model_pwg.state_dict())
    model_pwg.eval()
    model_espnet2.eval()

    with torch.no_grad():
        c = torch.randn(3, 5)
        out_pwg = model_pwg.inference(c)
        out_espnet2 = model_espnet2.inference(c)
        np.testing.assert_array_equal(
            out_pwg.cpu().numpy(), out_espnet2.cpu().numpy(),
        )

# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Test code for MelGAN modules."""

import numpy as np
import pytest
import torch

from espnet2.gan_tts.hifigan.loss import (
    DiscriminatorAdversarialLoss,
    FeatureMatchLoss,
    GeneratorAdversarialLoss,
)
from espnet2.gan_tts.melgan import MelGANGenerator, MelGANMultiScaleDiscriminator


def make_melgan_generator_args(**kwargs):
    defaults = dict(
        in_channels=80,
        out_channels=1,
        kernel_size=7,
        channels=32,
        bias=True,
        upsample_scales=[4, 4],
        stack_kernel_size=3,
        stacks=2,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.2},
        pad="ReflectionPad1d",
        pad_params={},
        use_final_nonlinear_activation=True,
        use_weight_norm=True,
    )
    defaults.update(kwargs)
    return defaults


def make_melgan_discriminator_args(**kwargs):
    defaults = dict(
        in_channels=1,
        out_channels=1,
        scales=2,
        downsample_pooling="AvgPool1d",
        # follow the official implementation setting
        downsample_pooling_params={
            "kernel_size": 4,
            "stride": 2,
            "padding": 1,
            "count_include_pad": False,
        },
        kernel_sizes=[5, 3],
        channels=16,
        max_downsample_channels=32,
        bias=True,
        downsample_scales=[2, 2],
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.2},
        pad="ReflectionPad1d",
        pad_params={},
        use_weight_norm=True,
    )
    defaults.update(kwargs)
    return defaults


@pytest.mark.skipif(
    "1.6" in torch.__version__,
    reason="group conv in pytorch 1.6 has an issue. "
    "See https://github.com/pytorch/pytorch/issues/42446.",
)
@pytest.mark.parametrize(
    "dict_g, dict_d",
    [
        ({}, {}),
        ({}, {"scales": 4}),
        ({}, {"kernel_sizes": [7, 5]}),
        ({}, {"max_downsample_channels": 128}),
        ({}, {"downsample_scales": [4, 4]}),
        ({}, {"pad": "ConstantPad1d", "pad_params": {"value": 0.0}}),
        ({}, {"nonlinear_activation": "ReLU", "nonlinear_activation_params": {}}),
    ],
)
def test_melgan_generator_and_discriminator(dict_g, dict_d):
    # setup
    batch_size = 2
    batch_length = 512
    args_g = make_melgan_generator_args(**dict_g)
    args_d = make_melgan_discriminator_args(**dict_d)
    y = torch.randn(batch_size, 1, batch_length)
    c = torch.randn(
        batch_size,
        args_g["in_channels"],
        batch_length // np.prod(args_g["upsample_scales"]),
    )
    model_g = MelGANGenerator(**args_g)
    model_d = MelGANMultiScaleDiscriminator(**args_d)
    gen_adv_criterion = GeneratorAdversarialLoss()
    dis_adv_criterion = DiscriminatorAdversarialLoss()
    feat_match_criterion = FeatureMatchLoss()
    optimizer_g = torch.optim.Adam(model_g.parameters())
    optimizer_d = torch.optim.Adam(model_d.parameters())

    # check generator trainable
    y_hat = model_g(c)
    p_hat = model_d(y_hat)
    adv_loss = gen_adv_criterion(p_hat)
    with torch.no_grad():
        p = model_d(y)
    fm_loss = feat_match_criterion(p_hat, p)
    loss_g = adv_loss + fm_loss
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
    try:
        from scipy.signal import kaiser  # NOQA
    except ImportError:
        pytest.skip("Kaiser window was not found at scipy.signal. Check scipy version.")
    from parallel_wavegan.models import MelGANGenerator as PWGMelGANGenerator

    model_pwg = PWGMelGANGenerator(**make_melgan_generator_args())
    model_espnet2 = MelGANGenerator(**make_melgan_generator_args())
    model_espnet2.load_state_dict(model_pwg.state_dict())
    model_pwg.eval()
    model_espnet2.eval()

    with torch.no_grad():
        c = torch.randn(5, 80)
        out_pwg = model_pwg.inference(c)
        out_espnet2 = model_espnet2.inference(c)
        np.testing.assert_array_equal(
            out_pwg.cpu().numpy(),
            out_espnet2.cpu().numpy(),
        )

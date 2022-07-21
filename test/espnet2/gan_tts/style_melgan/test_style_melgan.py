# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Test code for StyleMelGAN modules."""

import numpy as np
import pytest
import torch

from espnet2.gan_tts.hifigan.loss import (
    DiscriminatorAdversarialLoss,
    GeneratorAdversarialLoss,
)
from espnet2.gan_tts.style_melgan import StyleMelGANDiscriminator, StyleMelGANGenerator


def make_style_melgan_generator_args(**kwargs):
    defaults = dict(
        in_channels=32,
        aux_channels=5,
        channels=16,
        out_channels=1,
        kernel_size=9,
        dilation=2,
        bias=True,
        noise_upsample_scales=[11, 2, 2, 2],
        noise_upsample_activation="LeakyReLU",
        noise_upsample_activation_params={"negative_slope": 0.2},
        upsample_scales=[2, 2],
        upsample_mode="nearest",
        gated_function="softmax",
        use_weight_norm=True,
    )
    defaults.update(kwargs)
    return defaults


def make_style_melgan_discriminator_args(**kwargs):
    defaults = dict(
        repeats=2,
        window_sizes=[128, 256],
        pqmf_params=[[1, None, None, None], [2, 62, 0.26700, 9.0],],
        discriminator_params={
            "out_channels": 1,
            "kernel_sizes": [5, 3],
            "channels": 16,
            "max_downsample_channels": 32,
            "bias": True,
            "downsample_scales": [4, 4],
            "nonlinear_activation": "LeakyReLU",
            "nonlinear_activation_params": {"negative_slope": 0.2},
            "pad": "ReflectionPad1d",
            "pad_params": {},
        },
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
    "dict_g, dict_d", [({}, {}), ({"gated_function": "sigmoid"}, {}),],
)
def test_style_melgan_trainable(dict_g, dict_d):
    # setup
    args_g = make_style_melgan_generator_args(**dict_g)
    args_d = make_style_melgan_discriminator_args(**dict_d)
    batch_size = 2
    batch_length = np.prod(args_g["noise_upsample_scales"]) * np.prod(
        args_g["upsample_scales"]
    )
    y = torch.randn(batch_size, 1, batch_length)
    c = torch.randn(
        batch_size,
        args_g["aux_channels"],
        batch_length // np.prod(args_g["upsample_scales"]),
    )
    model_g = StyleMelGANGenerator(**args_g)
    model_d = StyleMelGANDiscriminator(**args_d)
    gen_adv_criterion = GeneratorAdversarialLoss()
    dis_adv_criterion = DiscriminatorAdversarialLoss()
    optimizer_g = torch.optim.Adam(model_g.parameters())
    optimizer_d = torch.optim.Adam(model_d.parameters())

    # check generator trainable
    y_hat = model_g(c)
    p_hat = model_d(y_hat)
    adv_loss = gen_adv_criterion(p_hat)
    loss_g = adv_loss
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
    from parallel_wavegan.models import StyleMelGANGenerator as PWGStyleMelGANGenerator

    model_pwg = PWGStyleMelGANGenerator(**make_style_melgan_generator_args())
    model_espnet2 = StyleMelGANGenerator(**make_style_melgan_generator_args())
    model_espnet2.load_state_dict(model_pwg.state_dict())
    model_pwg.eval()
    model_espnet2.eval()

    with torch.no_grad():
        c = torch.randn(3, 5)
        torch.manual_seed(1)
        out_pwg = model_pwg.inference(c)
        torch.manual_seed(1)
        out_espnet2 = model_espnet2.inference(c)
        np.testing.assert_array_equal(
            out_pwg.cpu().numpy(), out_espnet2.cpu().numpy(),
        )

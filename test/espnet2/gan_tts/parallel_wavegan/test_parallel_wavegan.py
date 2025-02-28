# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Test code for ParallelWaveGAN modules."""

import numpy as np
import pytest
import torch

from espnet2.gan_tts.hifigan.loss import (
    DiscriminatorAdversarialLoss,
    GeneratorAdversarialLoss,
)
from espnet2.gan_tts.parallel_wavegan import (
    ParallelWaveGANDiscriminator,
    ParallelWaveGANGenerator,
)


def make_generator_args(**kwargs):
    defaults = dict(
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        layers=6,
        stacks=3,
        residual_channels=8,
        gate_channels=16,
        skip_channels=8,
        aux_channels=10,
        aux_context_window=0,
        use_weight_norm=True,
        upsample_conditional_features=True,
        upsample_net="ConvInUpsampleNetwork",
        upsample_params={"upsample_scales": [4, 4]},
    )
    defaults.update(kwargs)
    return defaults


def make_discriminator_args(**kwargs):
    defaults = dict(
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        layers=5,
        conv_channels=16,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.2},
        bias=True,
        use_weight_norm=True,
    )
    defaults.update(kwargs)
    return defaults


@pytest.mark.parametrize(
    "dict_g, dict_d",
    [
        ({}, {}),
        ({"layers": 1, "stacks": 1}, {}),
        ({}, {"layers": 1}),
        ({"kernel_size": 5}, {}),
        ({}, {"kernel_size": 5}),
        ({"gate_channels": 8}, {}),
        ({"stacks": 1}, {}),
        ({"use_weight_norm": False}, {"use_weight_norm": False}),
        ({"aux_context_window": 2}, {}),
        ({"upsample_net": "UpsampleNetwork"}, {}),
        (
            {"upsample_params": {"upsample_scales": [4], "freq_axis_kernel_size": 3}},
            {},
        ),
        (
            {
                "upsample_params": {
                    "upsample_scales": [4],
                    "nonlinear_activation": "ReLU",
                }
            },
            {},
        ),
        (
            {
                "upsample_conditional_features": False,
                "upsample_params": {"upsample_scales": [1]},
            },
            {},
        ),
    ],
)
def test_parallel_wavegan_generator_and_discriminator(dict_g, dict_d):
    # setup
    batch_size = 4
    batch_length = 4096
    args_g = make_generator_args(**dict_g)
    args_d = make_discriminator_args(**dict_d)
    y = torch.randn(batch_size, 1, batch_length)
    c = torch.randn(
        batch_size,
        args_g["aux_channels"],
        batch_length // np.prod(args_g["upsample_params"]["upsample_scales"]),
    )
    model_g = ParallelWaveGANGenerator(**args_g)
    model_d = ParallelWaveGANDiscriminator(**args_d)
    gen_adv_criterion = GeneratorAdversarialLoss()
    dis_adv_criterion = DiscriminatorAdversarialLoss()
    optimizer_g = torch.optim.Adam(model_g.parameters())
    optimizer_d = torch.optim.Adam(model_d.parameters())

    # check generator trainable
    y_hat = model_g(c)
    p_hat = model_d(y_hat)
    loss_g = gen_adv_criterion(p_hat)
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


@pytest.mark.execution_timeout(10)
@pytest.mark.skipif(
    not is_parallel_wavegan_available, reason="parallel_wavegan is not installed."
)
def test_parallel_wavegan_compatibility():
    try:
        from scipy.signal import kaiser  # NOQA
    except ImportError:
        pytest.skip("Kaiser window was not found at scipy.signal. Check scipy version.")
    from parallel_wavegan.models import (
        ParallelWaveGANGenerator as PWGParallelWaveGANGenerator,
    )

    model_pwg = PWGParallelWaveGANGenerator(**make_generator_args())
    model_espnet2 = ParallelWaveGANGenerator(**make_generator_args())
    model_espnet2.load_state_dict(model_pwg.state_dict())
    model_pwg.eval()
    model_espnet2.eval()
    # NOTE(kan-bayashi): Use float64 to avoid numerical error in CI
    dtype = torch.float64
    model_pwg.to(dtype=dtype)
    model_espnet2.to(dtype=dtype)

    with torch.no_grad():
        z = torch.randn(3 * 16, 1, dtype=dtype)
        c = torch.randn(3, 10, dtype=dtype)
        out_pwg = model_pwg.inference(c, z)
        out_espnet2 = model_espnet2.inference(c, z)
        np.testing.assert_allclose(
            out_pwg.cpu().numpy(),
            out_espnet2.cpu().numpy(),
            rtol=1e-5,
        )

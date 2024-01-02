import pytest
import torch
from packaging.version import parse as V
from torch import Tensor

from espnet2.enh.diffusion.score_based_diffusion import ScoreModel

is_torch_1_12_1_plus = V(torch.__version__) >= V("1.12.1")


@pytest.mark.skipif(not is_torch_1_12_1_plus, reason="torch.complex32 is used")
def test_score_based_diffusion_forward_backward_dcunet():
    parameters = {
        "score_model": "dcunet",
        "score_model_conf": {
            "dcunet_architecture": "DCUNet-10",
        },
        "sde": "ouve",
        "sde_conf": {},
    }
    model = ScoreModel(**parameters)
    model.train()

    real = torch.rand(2, 128, 257)
    imag = torch.rand(2, 128, 257)
    feature_mix = real + 1j * imag
    real = torch.rand(2, 128, 257)
    imag = torch.rand(2, 128, 257)
    feature_ref = real + 1j * imag

    loss = model(feature_ref, feature_mix)
    loss.abs().mean().backward()


@pytest.mark.skipif(not is_torch_1_12_1_plus, reason="torch.complex32 is used")
def test_score_based_diffusion_forward_backward_ncsnpp():
    parameters = {
        "score_model": "ncsnpp",
        "score_model_conf": {
            "nf": 4,
            "ch_mult": (1, 1, 1),
        },
        "sde": "ouve",
        "sde_conf": {},
    }
    model = ScoreModel(**parameters)
    model.train()

    real = torch.rand(2, 64, 256)
    imag = torch.rand(2, 64, 256)
    feature_mix = real + 1j * imag
    real = torch.rand(2, 64, 256)
    imag = torch.rand(2, 64, 256)
    feature_ref = real + 1j * imag

    loss = model(feature_ref, feature_mix)
    loss.abs().mean().backward()


@pytest.mark.execution_timeout(20)
@pytest.mark.parametrize("sampler_type", ['pc', 'ode'])
@pytest.mark.skipif(not is_torch_1_12_1_plus, reason="torch.complex32 is used")
def test_score_based_diffusion_sampling(sampler_type):
    parameters = {
        "score_model": "ncsnpp",
        "score_model_conf": {
            "nf": 4,
            "ch_mult": (1, 1, 1),
        },
        "sde": "ouve",
        "sde_conf": {},
    }
    model = ScoreModel(**parameters)
    model.eval()

    real = torch.rand(2, 64, 256)
    imag = torch.rand(2, 64, 256)
    noise = real + 1j * imag

    output = model.enhance(noise, N=2, sampler_type=sampler_type)

    assert output.shape == noise.shape

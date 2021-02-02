import pytest

import torch
from torch import Tensor

from espnet2.enh.separator.asteroid_models import AsteroidModel_Converter


@pytest.mark.parametrize("encoder_output_dim", [1])
@pytest.mark.parametrize("model_name", ["ConvTasNet"])
@pytest.mark.parametrize("num_spk", [2])
@pytest.mark.parametrize("pretrained_path", ["mpariente/ConvTasNet_WHAM!_sepclean"])
@pytest.mark.parametrize("loss_type", ["si_snr"])
def test_asteroid_separator_pretrained_forward_backward_real(
    encoder_output_dim: int,
    model_name: str,
    num_spk: int,
    pretrained_path: str = "",
    loss_type: str = "si_snr",
):
    model = AsteroidModel_Converter(
        model_name=model_name,
        n_src=num_spk,
        loss_type=loss_type,
        pretrained_path=pretrained_path,
    )
    model.train()

    x = torch.rand(2, 50)
    x_lens = torch.tensor([50, 40], dtype=torch.long)

    est_wavs, flens, others = model(x, ilens=x_lens)

    assert isinstance(est_wavs[0], Tensor)
    assert len(est_wavs) == num_spk

    est_wavs[0].abs().mean().backward()


def test_asteroid_separator_invalid_type():
    with pytest.raises(ValueError):
        AsteroidModel_Converter(input_dim=2, model_name="ConvTasNet", num_spk=2)

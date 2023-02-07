import pytest
import torch
from torch import Tensor

from espnet2.enh.separator.fasnet_separator import FaSNetSeparator


@pytest.mark.parametrize("input_dim", [1])
@pytest.mark.parametrize("enc_dim", [4])
@pytest.mark.parametrize("feature_dim", [4])
@pytest.mark.parametrize("hidden_dim", [4])
@pytest.mark.parametrize("segment_size", [2])
@pytest.mark.parametrize("layer", [1, 2])
@pytest.mark.parametrize("num_spk", [1, 2])
@pytest.mark.parametrize("win_len", [2, 4])
@pytest.mark.parametrize("context_len", [2, 4])
@pytest.mark.parametrize("fasnet_type", ["fasnet", "ifasnet"])
@pytest.mark.parametrize("sr", [100])
def test_fasnet_separator_forward_backward_real(
    input_dim,
    enc_dim,
    feature_dim,
    hidden_dim,
    segment_size,
    layer,
    num_spk,
    win_len,
    context_len,
    fasnet_type,
    sr,
):
    model = FaSNetSeparator(
        input_dim=input_dim,
        enc_dim=enc_dim,
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        segment_size=segment_size,
        layer=layer,
        num_spk=num_spk,
        win_len=win_len,
        context_len=context_len,
        fasnet_type=fasnet_type,
        sr=sr,
    )
    model.train()

    x = torch.rand(2, 400, 4)
    x_lens = torch.tensor([400, 300], dtype=torch.long)

    separated, flens, others = model(x, ilens=x_lens)

    assert isinstance(separated[0], Tensor)
    assert len(separated) == num_spk

    separated[0].abs().mean().backward()


@pytest.mark.parametrize("fasnet_type", ["fasnet", "ifasnet"])
def test_fasnet_separator_output(fasnet_type):
    x = torch.rand(2, 800, 4)
    x_lens = torch.tensor([10, 8], dtype=torch.long)

    for num_spk in range(1, 3):
        model = FaSNetSeparator(
            input_dim=16,
            enc_dim=16,
            feature_dim=16,
            hidden_dim=16,
            segment_size=4,
            layer=2,
            num_spk=num_spk,
            win_len=2,
            context_len=2,
            fasnet_type=fasnet_type,
            sr=100,
        )
        model.eval()
        specs, _, others = model(x, x_lens)
        assert isinstance(specs, list)
        assert isinstance(others, dict)
        assert x[:, :, 0].shape == specs[0].shape

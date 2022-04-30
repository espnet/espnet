import pytest

import torch
from torch import Tensor

from espnet2.enh.separator.svoice_separator import SVoiceSeparator


@pytest.mark.parametrize("input_dim", [1])
@pytest.mark.parametrize("enc_dim", [4])
@pytest.mark.parametrize("kernel_size", [4])
@pytest.mark.parametrize("hidden_size", [4])
@pytest.mark.parametrize("num_spk", [1, 2])
@pytest.mark.parametrize("num_layers", [1, 2])
@pytest.mark.parametrize("segment_size", [2])
@pytest.mark.parametrize("bidirectional", [False])
@pytest.mark.parametrize("input_normalize", [False])
def test_svoice_separator_forward_backward(
    input_dim,
    enc_dim,
    kernel_size,
    hidden_size,
    num_spk,
    num_layers,
    segment_size,
    bidirectional,
    input_normalize,
):
    model = SVoiceSeparator(
        input_dim=input_dim,
        enc_dim=enc_dim,
        kernel_size=kernel_size,
        hidden_size=hidden_size,
        num_spk=num_spk,
        num_layers=num_layers,
        segment_size=segment_size,
        bidirectional=bidirectional,
        input_normalize=input_normalize,
    )
    model.train()

    x = torch.rand(2, 800)
    x_lens = torch.tensor([400, 300], dtype=torch.long)

    separated, _, _ = model(x, ilens=x_lens)

    assert isinstance(separated[0][0], Tensor)
    assert len(separated) == num_layers

    separated[0][0].mean().backward()


def test_svoice_separator_output_train():
    x = torch.rand(2, 800)
    x_lens = torch.tensor([10, 8], dtype=torch.long)

    for num_spk in range(1, 3):
        model = SVoiceSeparator(
            input_dim=12,
            enc_dim=8,
            kernel_size=8,
            hidden_size=8,
            num_spk=num_spk,
            num_layers=4,
            segment_size=2,
            bidirectional=False,
            input_normalize=False,
        )
        model.train()
        waveforms, _, _ = model(x, x_lens)
        assert isinstance(waveforms, list)
        assert isinstance(waveforms[0], list)
        assert x[0].shape == waveforms[0][0][0].shape


def test_svoice_separator_output_eval():
    x = torch.rand(2, 800)
    x_lens = torch.tensor([10, 8], dtype=torch.long)

    for num_spk in range(1, 3):
        model = SVoiceSeparator(
            input_dim=12,
            enc_dim=8,
            kernel_size=8,
            hidden_size=8,
            num_spk=num_spk,
            num_layers=4,
            segment_size=2,
            bidirectional=False,
            input_normalize=False,
        )
        model.eval()
        waveforms, _, _ = model(x, x_lens)
        assert isinstance(waveforms, list)
        assert x[0].shape == waveforms[0][0].shape

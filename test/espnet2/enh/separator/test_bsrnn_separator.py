import pytest
import torch
from torch_complex import ComplexTensor

from espnet2.enh.separator.bsrnn_separator import BSRNNSeparator


@pytest.mark.parametrize("input_dim", [481])
@pytest.mark.parametrize("num_spk", [1, 2])
@pytest.mark.parametrize("num_channels", [16])
@pytest.mark.parametrize("num_layers", [3])
@pytest.mark.parametrize("target_fs", [48000])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("norm_type", ["cfLN", "cLN", "BN", "GN"])
def test_bsrnn_separator_forward_backward_complex(
    input_dim,
    num_spk,
    num_channels,
    num_layers,
    target_fs,
    causal,
    norm_type,
):
    model = BSRNNSeparator(
        input_dim=input_dim,
        num_spk=num_spk,
        num_channels=num_channels,
        num_layers=num_layers,
        target_fs=target_fs,
        causal=causal,
        norm_type=norm_type,
    )
    model.train()

    real = torch.rand(2, 10, input_dim)
    imag = torch.rand(2, 10, input_dim)
    x = ComplexTensor(real, imag)
    x_lens = torch.tensor([10, 8], dtype=torch.long)

    masked, flens, others = model(x, ilens=x_lens)

    assert isinstance(masked[0], ComplexTensor)
    assert len(masked) == num_spk

    masked[0].abs().mean().backward()


@pytest.mark.parametrize("input_dim", [481])
@pytest.mark.parametrize("num_spk", [1, 2])
@pytest.mark.parametrize("num_channels", [16])
@pytest.mark.parametrize("num_layers", [3])
@pytest.mark.parametrize("target_fs", [48000])
@pytest.mark.parametrize("causal", [True, False])
def test_bsrnn_separator_forward_backward_real(
    input_dim,
    num_spk,
    num_channels,
    num_layers,
    target_fs,
    causal,
):
    model = BSRNNSeparator(
        input_dim=input_dim,
        num_spk=num_spk,
        num_channels=num_channels,
        num_layers=num_layers,
        target_fs=target_fs,
        causal=causal,
    )
    model.train()

    x = torch.randn(2, 10, input_dim, 2)
    x_lens = torch.tensor([10, 8], dtype=torch.long)

    masked, flens, others = model(x, ilens=x_lens)

    assert len(masked) == num_spk

    masked[0].abs().mean().backward()


def test_bsrnn_separator_with_different_sf():
    x_lens = torch.tensor([10, 8], dtype=torch.long)

    model = BSRNNSeparator(
        input_dim=481,
        num_spk=1,
        num_channels=10,
        num_layers=3,
        target_fs=48000,
        causal=True,
    )
    model.eval()

    for sf in (8000, 16000, 48000):
        f = int(sf * 0.01) + 1
        x = torch.randn(2, 10, f, 2)
        model(x, ilens=x_lens)


@pytest.mark.parametrize("fs", [8000, 16000, 24000, 32000, 44100, 48000])
def test_bsrnn_separator_with_fs_arg(fs):
    x_lens = torch.tensor([10, 8], dtype=torch.long)

    model = BSRNNSeparator(
        input_dim=481,
        num_spk=1,
        num_channels=10,
        num_layers=3,
        target_fs=48000,
        causal=True,
    )
    model.eval()

    f = int(fs * 0.01) + 1
    x = torch.randn(2, 10, f, 2)
    y1 = model(x, ilens=x_lens)[0]
    y2 = model(x, ilens=x_lens, additional={"fs": fs})[0]
    for yy1, yy2 in zip(y1, y2):
        torch.testing.assert_close(yy1.real, yy2.real)
        torch.testing.assert_close(yy1.imag, yy2.imag)

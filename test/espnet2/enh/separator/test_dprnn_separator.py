import pytest
import torch
from torch import Tensor
from torch_complex import ComplexTensor

from espnet2.enh.separator.dprnn_separator import DPRNNSeparator


@pytest.mark.parametrize("input_dim", [5])
@pytest.mark.parametrize("rnn_type", ["lstm", "gru"])
@pytest.mark.parametrize("layer", [1, 3])
@pytest.mark.parametrize("bidirectional", [True, False])
@pytest.mark.parametrize("unit", [8])
@pytest.mark.parametrize("dropout", [0.0, 0.2])
@pytest.mark.parametrize("num_spk", [1, 2])
@pytest.mark.parametrize("nonlinear", ["relu", "sigmoid", "tanh"])
@pytest.mark.parametrize("segment_size", [2, 4])
def test_dprnn_separator_forward_backward_complex(
    input_dim,
    rnn_type,
    bidirectional,
    layer,
    unit,
    dropout,
    num_spk,
    nonlinear,
    segment_size,
):
    model = DPRNNSeparator(
        input_dim=input_dim,
        rnn_type=rnn_type,
        bidirectional=bidirectional,
        num_spk=num_spk,
        nonlinear=nonlinear,
        layer=layer,
        unit=unit,
        segment_size=segment_size,
        dropout=dropout,
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


@pytest.mark.parametrize("input_dim", [5])
@pytest.mark.parametrize("rnn_type", ["lstm", "gru"])
@pytest.mark.parametrize("layer", [1, 3])
@pytest.mark.parametrize("bidirectional", [True, False])
@pytest.mark.parametrize("unit", [8])
@pytest.mark.parametrize("dropout", [0.0, 0.2])
@pytest.mark.parametrize("num_spk", [1, 2])
@pytest.mark.parametrize("nonlinear", ["relu", "sigmoid", "tanh"])
@pytest.mark.parametrize("segment_size", [2, 4])
def test_dprnn_separator_forward_backward_real(
    input_dim,
    rnn_type,
    bidirectional,
    layer,
    unit,
    dropout,
    num_spk,
    nonlinear,
    segment_size,
):
    model = DPRNNSeparator(
        input_dim=input_dim,
        rnn_type=rnn_type,
        bidirectional=bidirectional,
        num_spk=num_spk,
        nonlinear=nonlinear,
        layer=layer,
        unit=unit,
        segment_size=segment_size,
        dropout=dropout,
    )
    model.train()

    x = torch.rand(2, 10, input_dim)
    x_lens = torch.tensor([10, 8], dtype=torch.long)

    masked, flens, others = model(x, ilens=x_lens)

    assert isinstance(masked[0], Tensor)
    assert len(masked) == num_spk

    masked[0].abs().mean().backward()


def test_dprnn_separator_invalid_type():
    with pytest.raises(ValueError):
        DPRNNSeparator(
            input_dim=10,
            rnn_type="rnn",
            layer=2,
            unit=10,
            dropout=0.1,
            num_spk=2,
            nonlinear="fff",
            segment_size=2,
        )


def test_dprnn_separator_output():
    x = torch.rand(2, 10, 10)
    x_lens = torch.tensor([10, 8], dtype=torch.long)

    for num_spk in range(1, 3):
        model = DPRNNSeparator(
            input_dim=10,
            rnn_type="rnn",
            layer=2,
            unit=10,
            dropout=0.1,
            num_spk=2,
            nonlinear="relu",
            segment_size=2,
        )
        model.eval()
        specs, _, others = model(x, x_lens)
        assert isinstance(specs, list)
        assert isinstance(others, dict)
        assert x.shape == specs[0].shape
        for n in range(num_spk):
            assert "mask_spk{}".format(n + 1) in others
            assert specs[n].shape == others["mask_spk{}".format(n + 1)].shape

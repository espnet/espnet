import pytest
import torch
from torch import Tensor
from torch_complex import ComplexTensor

from espnet2.enh.separator.dan_separator import DANSeparator


@pytest.mark.parametrize("input_dim", [5])
@pytest.mark.parametrize("rnn_type", ["blstm"])
@pytest.mark.parametrize("layer", [1, 3])
@pytest.mark.parametrize("unit", [8])
@pytest.mark.parametrize("dropout", [0.0, 0.2])
@pytest.mark.parametrize("num_spk", [2])
@pytest.mark.parametrize("emb_D", [40])
@pytest.mark.parametrize("nonlinear", ["relu", "sigmoid", "tanh"])
def test_dan_separator_forward_backward_complex(
    input_dim, rnn_type, layer, unit, dropout, num_spk, emb_D, nonlinear
):
    model = DANSeparator(
        input_dim=input_dim,
        rnn_type=rnn_type,
        layer=layer,
        unit=unit,
        dropout=dropout,
        num_spk=num_spk,
        emb_D=emb_D,
        nonlinear=nonlinear,
    )
    model.train()

    real = torch.rand(2, 10, input_dim)
    imag = torch.rand(2, 10, input_dim)
    x = ComplexTensor(real, imag)
    x_lens = torch.tensor([10, 8], dtype=torch.long)

    o = []
    for i in range(num_spk):
        o.append(ComplexTensor(real, imag))

    sep_others = {}
    sep_others["feature_ref"] = o

    masked, flens, others = model(x, ilens=x_lens, additional=sep_others)

    assert isinstance(masked[0], ComplexTensor)
    assert len(masked) == num_spk

    masked[0].abs().mean().backward()


@pytest.mark.parametrize("input_dim", [5])
@pytest.mark.parametrize("rnn_type", ["blstm"])
@pytest.mark.parametrize("layer", [1, 3])
@pytest.mark.parametrize("unit", [8])
@pytest.mark.parametrize("dropout", [0.0, 0.2])
@pytest.mark.parametrize("num_spk", [1, 2])
@pytest.mark.parametrize("emb_D", [40])
@pytest.mark.parametrize("nonlinear", ["relu", "sigmoid", "tanh"])
def test_dan_separator_forward_backward_real(
    input_dim, rnn_type, layer, unit, dropout, num_spk, emb_D, nonlinear
):
    model = DANSeparator(
        input_dim=input_dim,
        rnn_type=rnn_type,
        layer=layer,
        unit=unit,
        dropout=dropout,
        num_spk=num_spk,
        emb_D=emb_D,
        nonlinear=nonlinear,
    )
    model.train()

    x = torch.rand(2, 10, input_dim)
    x_lens = torch.tensor([10, 8], dtype=torch.long)

    o = []
    for i in range(num_spk):
        o.append(ComplexTensor(x, x))

    sep_others = {}
    sep_others["feature_ref"] = o

    masked, flens, others = model(x, ilens=x_lens, additional=sep_others)

    assert isinstance(masked[0], Tensor)
    assert len(masked) == num_spk

    masked[0].abs().mean().backward()


def test_dan_separator_invalid_type():
    with pytest.raises(ValueError):
        DANSeparator(
            input_dim=10,
            rnn_type="rnn",
            layer=2,
            unit=10,
            dropout=0.1,
            num_spk=2,
            emb_D=40,
            nonlinear="fff",
        )


def test_dan_separator_output():

    x = torch.rand(1, 10, 10)
    x_lens = torch.tensor([10], dtype=torch.long)

    for num_spk in range(1, 4):
        model = DANSeparator(
            input_dim=10,
            rnn_type="rnn",
            layer=2,
            unit=10,
            dropout=0.1,
            num_spk=num_spk,
            emb_D=40,
            nonlinear="relu",
        )
        model.eval()
        specs, _, others = model(x, x_lens)
        assert isinstance(specs, list)
        assert isinstance(others, dict)
        for n in range(num_spk):
            assert "mask_spk{}".format(n + 1) in others
            assert specs[n].shape == others["mask_spk{}".format(n + 1)].shape

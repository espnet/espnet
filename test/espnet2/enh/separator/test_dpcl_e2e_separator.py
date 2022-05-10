import pytest

import torch
from torch import Tensor
from torch_complex import ComplexTensor

from espnet2.enh.separator.dpcl_e2e_separator import DPCLE2ESeparator


@pytest.mark.parametrize("input_dim", [5])
@pytest.mark.parametrize("rnn_type", ["blstm"])
@pytest.mark.parametrize("layer", [1, 3])
@pytest.mark.parametrize("unit", [8])
@pytest.mark.parametrize("dropout", [0.0, 0.2])
@pytest.mark.parametrize("num_spk", [2])
@pytest.mark.parametrize("emb_D", [40])
@pytest.mark.parametrize("nonlinear", ["relu", "sigmoid", "tanh"])
@pytest.mark.parametrize("alpha", [1.0, 5.0])
@pytest.mark.parametrize("max_iteration", [100, 500])
def test_dpcl_e2e_separator_forward_backward_complex(
    input_dim,
    rnn_type,
    layer,
    unit,
    dropout,
    num_spk,
    emb_D,
    nonlinear,
    alpha,
    max_iteration,
):
    model = DPCLE2ESeparator(
        input_dim=input_dim,
        rnn_type=rnn_type,
        layer=layer,
        unit=unit,
        dropout=dropout,
        num_spk=num_spk,
        emb_D=emb_D,
        nonlinear=nonlinear,
        alpha=alpha,
        max_iteration=max_iteration,
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
@pytest.mark.parametrize("rnn_type", ["blstm"])
@pytest.mark.parametrize("layer", [1, 3])
@pytest.mark.parametrize("unit", [8])
@pytest.mark.parametrize("dropout", [0.0, 0.2])
@pytest.mark.parametrize("num_spk", [2])
@pytest.mark.parametrize("emb_D", [40])
@pytest.mark.parametrize("nonlinear", ["relu", "sigmoid", "tanh"])
@pytest.mark.parametrize("alpha", [1.0, 5.0])
@pytest.mark.parametrize("max_iteration", [100, 500])
def test_dpcl_e2e_separator_forward_backward_real(
    input_dim,
    rnn_type,
    layer,
    unit,
    dropout,
    num_spk,
    emb_D,
    nonlinear,
    alpha,
    max_iteration,
):
    model = DPCLE2ESeparator(
        input_dim=input_dim,
        rnn_type=rnn_type,
        layer=layer,
        unit=unit,
        dropout=dropout,
        num_spk=num_spk,
        emb_D=emb_D,
        nonlinear=nonlinear,
        alpha=alpha,
        max_iteration=max_iteration,
    )
    model.train()

    x = torch.rand(2, 10, input_dim)
    x_lens = torch.tensor([10, 8], dtype=torch.long)

    masked, flens, others = model(x, ilens=x_lens)

    assert isinstance(masked[0], Tensor)
    assert len(masked) == num_spk

    masked[0].abs().mean().backward()


def test_dpcl_e2e_separator_invalid_type():
    with pytest.raises(ValueError):
        DPCLE2ESeparator(
            input_dim=10,
            rnn_type="rnn",
            layer=2,
            unit=10,
            dropout=0.1,
            num_spk=2,
            emb_D=40,
            nonlinear="fff",
            alpha=5.0,
            max_iteration=100,
        )


def test_dpcl_e2e_separator_output():

    x = torch.rand(1, 10, 10)
    x_lens = torch.tensor([10], dtype=torch.long)

    for num_spk in range(1, 4):
        model = DPCLE2ESeparator(
            input_dim=10,
            rnn_type="rnn",
            layer=2,
            unit=10,
            dropout=0.1,
            num_spk=num_spk,
            emb_D=40,
            nonlinear="relu",
            alpha=5.0,
            max_iteration=100,
        )
        model.eval()
        specs, _, others = model(x, x_lens)
        assert isinstance(specs, list)
        assert isinstance(others, dict)
        for n in range(num_spk):
            assert "mask_spk{}".format(n + 1) in others
            assert specs[n].shape == others["mask_spk{}".format(n + 1)].shape

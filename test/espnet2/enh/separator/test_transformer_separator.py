import pytest
import torch
from torch import Tensor
from torch_complex import ComplexTensor

from espnet2.enh.separator.transformer_separator import TransformerSeparator


@pytest.mark.parametrize("input_dim", [5])
@pytest.mark.parametrize("num_spk", [1, 2])
@pytest.mark.parametrize("adim", [8])
@pytest.mark.parametrize("layers", [1, 3])
@pytest.mark.parametrize("aheads", [2])
@pytest.mark.parametrize("linear_units", [10])
@pytest.mark.parametrize("positionwise_layer_type", ["linear"])
@pytest.mark.parametrize("normalize_before", [True])
@pytest.mark.parametrize("concat_after", [True])
@pytest.mark.parametrize("use_scaled_pos_enc", [True])
@pytest.mark.parametrize("dropout_rate", [0.1])
@pytest.mark.parametrize("positional_dropout_rate", [0.1])
@pytest.mark.parametrize("attention_dropout_rate", [0.1])
@pytest.mark.parametrize("nonlinear", ["relu", "sigmoid", "tanh"])
def test_transformer_separator_forward_backward_complex(
    input_dim,
    adim,
    layers,
    aheads,
    linear_units,
    num_spk,
    nonlinear,
    positionwise_layer_type,
    normalize_before,
    concat_after,
    dropout_rate,
    positional_dropout_rate,
    attention_dropout_rate,
    use_scaled_pos_enc,
):
    model = TransformerSeparator(
        input_dim=input_dim,
        num_spk=num_spk,
        adim=adim,
        aheads=aheads,
        layers=layers,
        linear_units=linear_units,
        positionwise_layer_type=positionwise_layer_type,
        normalize_before=normalize_before,
        concat_after=concat_after,
        dropout_rate=dropout_rate,
        positional_dropout_rate=positional_dropout_rate,
        attention_dropout_rate=attention_dropout_rate,
        use_scaled_pos_enc=use_scaled_pos_enc,
        nonlinear=nonlinear,
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
@pytest.mark.parametrize("num_spk", [1, 2])
@pytest.mark.parametrize("adim", [8])
@pytest.mark.parametrize("layers", [1, 3])
@pytest.mark.parametrize("aheads", [2])
@pytest.mark.parametrize("linear_units", [10])
@pytest.mark.parametrize("positionwise_layer_type", ["linear"])
@pytest.mark.parametrize("normalize_before", [True])
@pytest.mark.parametrize("concat_after", [True])
@pytest.mark.parametrize("use_scaled_pos_enc", [True])
@pytest.mark.parametrize("dropout_rate", [0.1])
@pytest.mark.parametrize("positional_dropout_rate", [0.1])
@pytest.mark.parametrize("attention_dropout_rate", [0.1])
@pytest.mark.parametrize("nonlinear", ["relu", "sigmoid", "tanh"])
def test_transformer_separator_forward_backward_real(
    input_dim,
    adim,
    layers,
    aheads,
    linear_units,
    num_spk,
    nonlinear,
    positionwise_layer_type,
    normalize_before,
    concat_after,
    dropout_rate,
    positional_dropout_rate,
    attention_dropout_rate,
    use_scaled_pos_enc,
):
    model = TransformerSeparator(
        input_dim=input_dim,
        num_spk=num_spk,
        adim=adim,
        aheads=aheads,
        layers=layers,
        linear_units=linear_units,
        positionwise_layer_type=positionwise_layer_type,
        normalize_before=normalize_before,
        concat_after=concat_after,
        dropout_rate=dropout_rate,
        positional_dropout_rate=positional_dropout_rate,
        attention_dropout_rate=attention_dropout_rate,
        use_scaled_pos_enc=use_scaled_pos_enc,
        nonlinear=nonlinear,
    )
    model.train()

    x = torch.rand(2, 10, input_dim)
    x_lens = torch.tensor([10, 8], dtype=torch.long)

    masked, flens, others = model(x, ilens=x_lens)

    assert isinstance(masked[0], Tensor)
    assert len(masked) == num_spk

    masked[0].abs().mean().backward()


def test_transformer_separator_invalid_type():
    with pytest.raises(ValueError):
        TransformerSeparator(
            input_dim=10,
            nonlinear="fff",
        )


def test_transformer_separator_output():

    x = torch.rand(2, 10, 10)
    x_lens = torch.tensor([10, 8], dtype=torch.long)

    for num_spk in range(1, 3):
        model = TransformerSeparator(
            input_dim=10,
            layers=2,
            adim=4,
            aheads=2,
            num_spk=num_spk,
            linear_units=10,
            nonlinear="relu",
        )
        model.eval()
        specs, _, others = model(x, x_lens)
        assert isinstance(specs, list)
        assert isinstance(others, dict)
        for n in range(num_spk):
            assert "mask_spk{}".format(n + 1) in others
            assert specs[n].shape == others["mask_spk{}".format(n + 1)].shape

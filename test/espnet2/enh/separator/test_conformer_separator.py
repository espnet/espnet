import pytest

import torch
from torch import Tensor
from torch_complex.tensor import ComplexTensor

from espnet2.enh.separator.conformer_separator import ConformerSeparator


@pytest.mark.parametrize("input_dim", [15])
@pytest.mark.parametrize("num_spk", [1, 2])
@pytest.mark.parametrize("adim", [8])
@pytest.mark.parametrize("layers", [1, 3])
@pytest.mark.parametrize("aheads", [2])
@pytest.mark.parametrize("linear_units", [100])
@pytest.mark.parametrize("positionwise_layer_type", ["conv1d"])
@pytest.mark.parametrize("positionwise_conv_kernel_size", [3])
@pytest.mark.parametrize("normalize_before", [True])
@pytest.mark.parametrize("concat_after", [True])
@pytest.mark.parametrize("dropout_rate", [0.1])
@pytest.mark.parametrize("input_layer", ["linear", "conv2d"])
@pytest.mark.parametrize("positional_dropout_rate", [0.1])
@pytest.mark.parametrize("attention_dropout_rate", [0.1])
@pytest.mark.parametrize("nonlinear", ["relu", "sigmoid", "tanh"])
@pytest.mark.parametrize("conformer_pos_enc_layer_type", ["rel_pos"])
@pytest.mark.parametrize("conformer_self_attn_layer_type", ["rel_selfattn"])
@pytest.mark.parametrize("conformer_activation_type", ["relu", "tanh"])
@pytest.mark.parametrize("use_macaron_style_in_conformer", [True])
@pytest.mark.parametrize("use_cnn_in_conformer", [True])
@pytest.mark.parametrize("conformer_enc_kernel_size", [3, 5, 7])
@pytest.mark.parametrize("padding_idx", [-1])
def test_conformer_separator_forward_backward_complex(
    input_dim,
    num_spk,
    adim,
    aheads,
    layers,
    linear_units,
    positionwise_layer_type,
    positionwise_conv_kernel_size,
    normalize_before,
    concat_after,
    dropout_rate,
    input_layer,
    positional_dropout_rate,
    attention_dropout_rate,
    nonlinear,
    conformer_pos_enc_layer_type,
    conformer_self_attn_layer_type,
    conformer_activation_type,
    use_macaron_style_in_conformer,
    use_cnn_in_conformer,
    conformer_enc_kernel_size,
    padding_idx,
):
    model = ConformerSeparator(
        input_dim=input_dim,
        num_spk=num_spk,
        adim=adim,
        aheads=aheads,
        layers=layers,
        linear_units=linear_units,
        dropout_rate=dropout_rate,
        positional_dropout_rate=positional_dropout_rate,
        attention_dropout_rate=attention_dropout_rate,
        input_layer=input_layer,
        normalize_before=normalize_before,
        concat_after=concat_after,
        positionwise_layer_type=positionwise_layer_type,
        positionwise_conv_kernel_size=positionwise_conv_kernel_size,
        use_macaron_style_in_conformer=use_macaron_style_in_conformer,
        nonlinear=nonlinear,
        conformer_pos_enc_layer_type=conformer_pos_enc_layer_type,
        conformer_self_attn_layer_type=conformer_self_attn_layer_type,
        conformer_activation_type=conformer_activation_type,
        use_cnn_in_conformer=use_cnn_in_conformer,
        conformer_enc_kernel_size=conformer_enc_kernel_size,
        padding_idx=padding_idx,
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


@pytest.mark.parametrize("input_dim", [15])
@pytest.mark.parametrize("num_spk", [1, 2])
@pytest.mark.parametrize("adim", [8])
@pytest.mark.parametrize("layers", [1, 3])
@pytest.mark.parametrize("aheads", [2])
@pytest.mark.parametrize("linear_units", [100])
@pytest.mark.parametrize("positionwise_layer_type", ["conv1d"])
@pytest.mark.parametrize("positionwise_conv_kernel_size", [3])
@pytest.mark.parametrize("normalize_before", [True])
@pytest.mark.parametrize("concat_after", [True])
@pytest.mark.parametrize("dropout_rate", [0.1])
@pytest.mark.parametrize("input_layer", ["linear", "conv2d"])
@pytest.mark.parametrize("positional_dropout_rate", [0.1])
@pytest.mark.parametrize("attention_dropout_rate", [0.1])
@pytest.mark.parametrize("nonlinear", ["relu", "sigmoid", "tanh"])
@pytest.mark.parametrize("conformer_pos_enc_layer_type", ["rel_pos"])
@pytest.mark.parametrize("conformer_self_attn_layer_type", ["rel_selfattn"])
@pytest.mark.parametrize("conformer_activation_type", ["relu", "tanh"])
@pytest.mark.parametrize("use_macaron_style_in_conformer", [True])
@pytest.mark.parametrize("use_cnn_in_conformer", [True])
@pytest.mark.parametrize("conformer_enc_kernel_size", [3, 5, 7])
@pytest.mark.parametrize("padding_idx", [-1])
def test_conformer_separator_forward_backward_real(
    input_dim,
    num_spk,
    adim,
    aheads,
    layers,
    linear_units,
    positionwise_layer_type,
    positionwise_conv_kernel_size,
    normalize_before,
    concat_after,
    dropout_rate,
    input_layer,
    positional_dropout_rate,
    attention_dropout_rate,
    nonlinear,
    conformer_pos_enc_layer_type,
    conformer_self_attn_layer_type,
    conformer_activation_type,
    use_macaron_style_in_conformer,
    use_cnn_in_conformer,
    conformer_enc_kernel_size,
    padding_idx,
):
    model = ConformerSeparator(
        input_dim=input_dim,
        num_spk=num_spk,
        adim=adim,
        aheads=aheads,
        layers=layers,
        linear_units=linear_units,
        dropout_rate=dropout_rate,
        positional_dropout_rate=positional_dropout_rate,
        attention_dropout_rate=attention_dropout_rate,
        input_layer=input_layer,
        normalize_before=normalize_before,
        concat_after=concat_after,
        positionwise_layer_type=positionwise_layer_type,
        positionwise_conv_kernel_size=positionwise_conv_kernel_size,
        use_macaron_style_in_conformer=use_macaron_style_in_conformer,
        nonlinear=nonlinear,
        conformer_pos_enc_layer_type=conformer_pos_enc_layer_type,
        conformer_self_attn_layer_type=conformer_self_attn_layer_type,
        conformer_activation_type=conformer_activation_type,
        use_cnn_in_conformer=use_cnn_in_conformer,
        conformer_enc_kernel_size=conformer_enc_kernel_size,
        padding_idx=padding_idx,
    )
    model.train()

    x = torch.rand(2, 10, input_dim)
    x_lens = torch.tensor([10, 8], dtype=torch.long)

    masked, flens, others = model(x, ilens=x_lens)

    assert isinstance(masked[0], Tensor)
    assert len(masked) == num_spk

    masked[0].abs().mean().backward()


def test_Conformer_separator_invalid_type():
    with pytest.raises(ValueError):
        ConformerSeparator(
            input_dim=10,
            nonlinear="fff",
        )


def test_Conformer_separator_output():

    x = torch.rand(2, 10, 10)
    x_lens = torch.tensor([10, 8], dtype=torch.long)

    for num_spk in range(1, 3):
        model = ConformerSeparator(
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

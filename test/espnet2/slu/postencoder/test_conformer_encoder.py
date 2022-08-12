import pytest
import torch

from espnet2.slu.postencoder.conformer_postencoder import ConformerPostEncoder


@pytest.mark.parametrize("input_layer", ["linear"])
@pytest.mark.parametrize("positionwise_layer_type", ["conv1d", "conv1d-linear"])
@pytest.mark.parametrize(
    "rel_pos_type, pos_enc_layer_type, selfattention_layer_type",
    [
        ("legacy", "abs_pos", "selfattn"),
        ("latest", "rel_pos", "rel_selfattn"),
        ("legacy", "rel_pos", "rel_selfattn"),
        ("legacy", "legacy_rel_pos", "legacy_rel_selfattn"),
    ],
)
def test_encoder_forward_backward(
    input_layer,
    positionwise_layer_type,
    rel_pos_type,
    pos_enc_layer_type,
    selfattention_layer_type,
):
    encoder = ConformerPostEncoder(
        20,
        output_size=2,
        attention_heads=2,
        linear_units=4,
        num_blocks=2,
        input_layer=input_layer,
        macaron_style=False,
        rel_pos_type=rel_pos_type,
        pos_enc_layer_type=pos_enc_layer_type,
        selfattention_layer_type=selfattention_layer_type,
        activation_type="swish",
        use_cnn_module=True,
        cnn_module_kernel=3,
        positionwise_layer_type=positionwise_layer_type,
    )
    x = torch.randn(2, 32, 2, requires_grad=True)
    x_lens = torch.LongTensor([32, 28])
    print(x.shape)
    print(x_lens.shape)
    y, _ = encoder(x, x_lens)
    y.sum().backward()


def test_encoder_invalid_layer_type():
    with pytest.raises(ValueError):
        ConformerPostEncoder(20, rel_pos_type="dummy")
    with pytest.raises(ValueError):
        ConformerPostEncoder(20, pos_enc_layer_type="dummy")
    with pytest.raises(ValueError):
        ConformerPostEncoder(
            20, pos_enc_layer_type="abc_pos", selfattention_layer_type="dummy"
        )


def test_encoder_invalid_rel_pos_combination():
    with pytest.raises(AssertionError):
        ConformerPostEncoder(
            20,
            rel_pos_type="latest",
            pos_enc_layer_type="legacy_rel_pos",
            selfattention_layer_type="legacy_rel_sselfattn",
        )
    with pytest.raises(AssertionError):
        ConformerPostEncoder(
            20,
            pos_enc_layer_type="rel_pos",
            selfattention_layer_type="legacy_rel_sselfattn",
        )
    with pytest.raises(AssertionError):
        ConformerPostEncoder(
            20,
            pos_enc_layer_type="legacy_rel_pos",
            selfattention_layer_type="rel_sselfattn",
        )


def test_encoder_output_size():
    encoder = ConformerPostEncoder(20, output_size=256)
    assert encoder.output_size() == 256


def test_encoder_invalid_type():
    with pytest.raises(ValueError):
        ConformerPostEncoder(20, input_layer="fff")

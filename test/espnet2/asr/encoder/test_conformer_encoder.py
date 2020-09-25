import pytest
import torch

from espnet2.asr.encoder.conformer_encoder import ConformerEncoder


@pytest.mark.parametrize(
    "input_layer", ["linear", "conv2d", "conv2d6", "conv2d8", "embed"]
)
@pytest.mark.parametrize("positionwise_layer_type", ["conv1d", "conv1d-linear"])
@pytest.mark.parametrize(
    "pos_enc_layer_type, selfattention_layer_type",
    [("abs_pos", "selfattn"), ("rel_pos", "rel_selfattn")],
)
def test_encoder_forward_backward(
    input_layer, positionwise_layer_type, pos_enc_layer_type, selfattention_layer_type
):
    encoder = ConformerEncoder(
        20,
        output_size=2,
        attention_heads=2,
        linear_units=4,
        num_blocks=2,
        input_layer=input_layer,
        macaron_style=False,
        pos_enc_layer_type=pos_enc_layer_type,
        selfattention_layer_type=selfattention_layer_type,
        activation_type="swish",
        use_cnn_module=True,
        cnn_module_kernel=3,
        positionwise_layer_type=positionwise_layer_type,
    )
    if input_layer == "embed":
        x = torch.randint(0, 10, [2, 32])
    else:
        x = torch.randn(2, 32, 20, requires_grad=True)
    x_lens = torch.LongTensor([32, 28])
    y, _, _ = encoder(x, x_lens)
    y.sum().backward()


def test_encoder_output_size():
    encoder = ConformerEncoder(20, output_size=256)
    assert encoder.output_size() == 256


def test_encoder_invalid_type():
    with pytest.raises(ValueError):
        ConformerEncoder(20, input_layer="fff")

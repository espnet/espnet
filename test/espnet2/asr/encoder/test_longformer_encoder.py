import pytest
import torch

from espnet2.asr.encoder.longformer_encoder import LongformerEncoder

pytest.importorskip("longformer")


@pytest.mark.parametrize(
    "input_layer", ["linear", "conv2d", "conv2d2", "conv2d6", "conv2d8", "embed"]
)
@pytest.mark.parametrize("positionwise_layer_type", ["conv1d", "conv1d-linear"])
@pytest.mark.parametrize(
    "rel_pos_type, pos_enc_layer_type, selfattention_layer_type",
    [
        ("legacy", "abs_pos", "lf_selfattn"),
    ],
)
def test_encoder_forward_backward(
    input_layer,
    positionwise_layer_type,
    rel_pos_type,
    pos_enc_layer_type,
    selfattention_layer_type,
):
    pytest.importorskip("longformer")
    encoder = LongformerEncoder(
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
        attention_windows=[10, 10],
        attention_dilation=[1, 1],
        attention_mode="sliding_chunks",
    )
    if input_layer == "embed":
        x = torch.randint(0, 10, [2, 32])
    else:
        x = torch.randn(2, 32, 20, requires_grad=True)
    x_lens = torch.LongTensor([32, 28])
    y, _, _ = encoder(x, x_lens)
    y.sum().backward()


def test_encoder_invalid_layer_type():
    pytest.importorskip("longformer")
    with pytest.raises(ValueError):
        LongformerEncoder(20, pos_enc_layer_type="abc_pos")
    with pytest.raises(ValueError):
        LongformerEncoder(20, pos_enc_layer_type="dummy")
    with pytest.raises(ValueError):
        LongformerEncoder(
            20, pos_enc_layer_type="abc_pos", selfattention_layer_type="dummy"
        )


def test_encoder_invalid_windows_parameter():
    pytest.importorskip("longformer")
    with pytest.raises(ValueError):
        LongformerEncoder(20, attention_windows=[1, 1], num_blocks=4)
    with pytest.raises(ValueError):
        LongformerEncoder(20, attention_dilation=[1, 1], num_blocks=4)


def test_encoder_output_size():
    pytest.importorskip("longformer")
    encoder = LongformerEncoder(20, output_size=256)
    assert encoder.output_size() == 256


def test_encoder_invalid_type():
    pytest.importorskip("longformer")
    with pytest.raises(ValueError):
        LongformerEncoder(20, input_layer="fff")

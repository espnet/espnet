import pytest
import torch

from espnet2.asr.encoder.custom_encoder import CustomEncoder


@pytest.mark.parametrize(
    "input_layer",
    [
        {"layer_type": "conv2d", "hidden_size": 8},
        {"layer_type": "conv2d6", "hidden_size": 8},
        {"layer_type": "conv2d8", "hidden_size": 8},
        {"layer_type": "embed", "hidden_size": 8},
        {"layer_type": "linear", "hidden_size": 8},
        {"layer_type": "vgg2l", "hidden_size": 8},
    ],
)
@pytest.mark.parametrize(
    "tdnn_layer",
    [
        None,
        {"layer_type": "tdnn", "input_size": 8, "output_size": 8, "context_size": 1},
    ],
)
@pytest.mark.parametrize(
    "body_layer",
    [
        {
            "layer_type": "conformer",
            "hidden_size": 8,
            "linear_units": 8,
            "attention_heads": 1,
        },
        {
            "layer_type": "transformer",
            "hidden_size": 8,
            "linear_units": 8,
            "attention_heads": 1,
        },
    ],
)
@pytest.mark.parametrize(
    "main_classes",
    [
        {
            "positional_encoding_type": "abs_pos",
            "positionwise_type": "linear",
            "self_attention_type": "self_attn",
        },
        {
            "positional_encoding_type": "scaled_abs_pos",
            "positionwise_type": "conv1d",
            "self_attention_type": "self_attn",
        },
        {
            "positional_encoding_type": "rel_pos",
            "positionwise_type": "conv1d-linear",
            "self_attention_type": "rel_self_attn",
        },
    ],
)
@pytest.mark.parametrize("repeat", [0, 2])
def test_Encoder_forward_backward(
    input_layer, tdnn_layer, body_layer, main_classes, repeat
):
    if tdnn_layer is None:
        architecture = [input_layer, body_layer]
    else:
        architecture = [input_layer, tdnn_layer, body_layer]
    encoder = CustomEncoder(20, architecture, **main_classes, repeat=repeat)

    if architecture[0]["layer_type"] == "embed":
        x = torch.randint(0, 20, [2, 20])
    else:
        x = torch.randn(2, 20, 20, requires_grad=True)
    x_lens = torch.LongTensor([20, 8])

    y, _, _ = encoder(x, x_lens)
    y.sum().backward()

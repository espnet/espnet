import pytest
import torch

from espnet2.slu.postencoder.transformer_postencoder import TransformerPostEncoder


@pytest.mark.parametrize("input_layer", ["linear", "None"])
@pytest.mark.parametrize("positionwise_layer_type", ["conv1d", "conv1d-linear"])
def test_Encoder_forward_backward(
    input_layer,
    positionwise_layer_type,
):
    encoder = TransformerPostEncoder(
        20,
        output_size=40,
        input_layer=input_layer,
        positionwise_layer_type=positionwise_layer_type,
    )
    x = torch.randn(2, 10, 20, requires_grad=True)
    x_lens = torch.LongTensor([10, 8])
    y, _ = encoder(x, x_lens)
    y.sum().backward()


def test_Encoder_output_size():
    encoder = TransformerPostEncoder(20, output_size=256)
    assert encoder.output_size() == 256


def test_Encoder_invalid_type():
    with pytest.raises(ValueError):
        TransformerPostEncoder(20, input_layer="fff")

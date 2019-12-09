import pytest
import torch

from espnet2.asr.encoder.transformer_encoder import Encoder


@pytest.mark.parametrize("input_layer", ["linear", "conv2d", "embed", None])
@pytest.mark.parametrize("positionwise_layer_type",
                         ["conv1d", "conv1d-linear"])
def test_Encoder_forward_backward(input_layer, positionwise_layer_type):
    encoder = Encoder(20, attention_dim=40, input_layer=input_layer,
                      positionwise_layer_type=positionwise_layer_type)
    if input_layer == "embed":
        x = torch.randint(0, 10, [2, 10])
    elif input_layer is None:
        x = torch.randn(2, 10, 40, requires_grad=True)
    else:
        x = torch.randn(2, 10, 20, requires_grad=True)
    x_lens = torch.LongTensor([10, 8])
    y, _, _ = encoder(x, x_lens)
    y.sum().backward()


def test_Encoder_out_dim():
    encoder = Encoder(20, attention_dim=256)
    assert encoder.out_dim() == 256


def test_Encoder_invalid_type():
    with pytest.raises(ValueError):
        Encoder(20, input_layer='fff')

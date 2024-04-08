import pytest
import torch

from espnet2.asr.encoder.transformer_encoder_multispkr import TransformerEncoder


@pytest.mark.parametrize("input_layer", ["conv2d"])
@pytest.mark.parametrize("positionwise_layer_type", ["conv1d"])
@pytest.mark.parametrize("num_inf", [1, 2, 3])
def test_Encoder_forward_backward(
    input_layer,
    positionwise_layer_type,
    num_inf,
):
    encoder = TransformerEncoder(
        20,
        output_size=40,
        input_layer=input_layer,
        positionwise_layer_type=positionwise_layer_type,
        num_blocks=1,
        num_blocks_sd=1,
        num_inf=num_inf,
    )
    if input_layer == "embed":
        x = torch.randint(0, 10, [2, 10])
    else:
        x = torch.randn(2, 10, 20, requires_grad=True)
    x_lens = torch.LongTensor([10, 8])
    y, _, _ = encoder(x, x_lens)
    assert y.shape[:2] == torch.Size((2, num_inf))
    y.sum().backward()


def test_Encoder_output_size():
    encoder = TransformerEncoder(
        20,
        output_size=256,
        num_blocks=1,
        num_blocks_sd=1,
        num_inf=2,
    )
    assert encoder.output_size() == 256


def test_Encoder_invalid_type():
    with pytest.raises(ValueError):
        TransformerEncoder(20, input_layer="fff")

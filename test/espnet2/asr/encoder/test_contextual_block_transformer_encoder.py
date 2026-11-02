import pytest
import torch

from espnet2.asr.encoder.contextual_block_transformer_encoder import (  # noqa: H301
    ContextualBlockTransformerEncoder,
)


@pytest.mark.parametrize("input_layer", ["linear", "conv2d", "embed", None])
@pytest.mark.parametrize("positionwise_layer_type", ["conv1d", "conv1d-linear"])
def test_Encoder_forward_backward(input_layer, positionwise_layer_type):
    encoder = ContextualBlockTransformerEncoder(
        20,
        output_size=40,
        input_layer=input_layer,
        positionwise_layer_type=positionwise_layer_type,
        block_size=4,
        hop_size=2,
        look_ahead=1,
    )
    if input_layer == "embed":
        x = torch.randint(0, 10, [2, 10])
    elif input_layer is None:
        x = torch.randn(2, 10, 40, requires_grad=True)
    else:
        x = torch.randn(2, 10, 20, requires_grad=True)
    x_lens = torch.LongTensor([10, 8])
    y, _, _ = encoder(x, x_lens)
    y.sum().backward()

    # case of shorter than block size
    encoder2 = ContextualBlockTransformerEncoder(
        20,
        output_size=40,
        input_layer=input_layer,
        positionwise_layer_type=positionwise_layer_type,
        block_size=16,
        hop_size=8,
        look_ahead=4,
    )
    y2, _, _ = encoder2(x, x_lens)
    y2.sum().backward()


def test_Encoder_output_size():
    encoder = ContextualBlockTransformerEncoder(20, output_size=256)
    assert encoder.output_size() == 256


def test_Encoder_invalid_type():
    with pytest.raises(ValueError):
        ContextualBlockTransformerEncoder(20, input_layer="fff")


def test_Encoder_olens_is_independent_of_batch_mates():
    enc = ContextualBlockTransformerEncoder(
        input_size=8,
        output_size=4,
        attention_heads=2,
        linear_units=4,
        num_blocks=1,
        input_layer="conv2d",
        block_size=0,
    ).eval()

    short, long_ = 288, 363
    feats = torch.randn(2, long_, 8)
    with torch.no_grad():
        _, olens_alone, _ = enc(feats[1:2, :short], torch.tensor([short]))
        _, olens_batch, _ = enc(feats, torch.tensor([long_, short]))
        _, olens_pair, _ = enc(feats[:, :short], torch.tensor([short, short]))

    assert olens_alone.tolist() == [71]
    assert olens_batch[1].item() == 71
    assert olens_pair.tolist() == [71, 71]

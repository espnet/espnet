import pytest
import torch

from espnet2.asr.encoder.contextual_block_conformer_encoder import (  # noqa: H301
    ContextualBlockConformerEncoder,
)


@pytest.mark.parametrize("input_layer", ["linear", "conv2d"])
def test_Encoder_forward_backward(input_layer):
    encoder = ContextualBlockConformerEncoder(
        20,
        output_size=40,
        input_layer=input_layer,
        block_size=4,
        hop_size=2,
        look_ahead=1,
        macaron_style=False,
        use_cnn_module=False,
    )
    x = torch.randn(2, 10, 20, requires_grad=True)
    x_lens = torch.LongTensor([10, 8])
    y, _, _ = encoder(x, x_lens)
    y.sum().backward()


def test_Encoder_olens_is_independent_of_batch_mates():
    """Short row keeps olen 71 alone, in a mixed batch, and in a short pair."""
    enc = ContextualBlockConformerEncoder(
        input_size=8,
        output_size=4,
        attention_heads=2,
        linear_units=4,
        num_blocks=1,
        input_layer="conv2d",
        block_size=0,
        macaron_style=False,
        use_cnn_module=False,
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

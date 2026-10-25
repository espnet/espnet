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


def test_use_sdpa_reaches_the_attention_and_matches_the_default():
    x = torch.randn(2, 10, 20)
    x_lens = torch.LongTensor([10, 8])

    def build(**kwargs):
        torch.manual_seed(0)
        return TransformerPostEncoder(
            20,
            output_size=32,
            attention_heads=2,
            linear_units=32,
            num_blocks=2,
            **kwargs,
        )

    # eval() so dropout does not make the two passes draw different masks.
    sdpa_encoder = build(use_sdpa=True).eval()
    attentions = [m for m in sdpa_encoder.modules() if hasattr(m, "use_sdpa")]
    assert attentions and all(m.use_sdpa for m in attentions)

    with torch.no_grad():
        default_out, _ = build().eval()(x, x_lens)
        sdpa_out, _ = sdpa_encoder(x, x_lens)
    assert default_out.shape == sdpa_out.shape
    assert torch.allclose(default_out, sdpa_out, atol=1e-5)

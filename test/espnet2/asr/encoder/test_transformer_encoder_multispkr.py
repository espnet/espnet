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


def test_use_sdpa_reaches_the_attention_and_matches_the_default():
    """`use_sdpa` has to be forwarded, and must not change the output.

    The flag reaches PyTorch's scaled_dot_product_attention; checking every
    module that carries it, rather than one attribute name, keeps this honest
    if a layer gains a second attention.
    """
    x = torch.randn(2, 20, 20)
    x_lens = torch.LongTensor([20, 16])

    def build(**kwargs):
        torch.manual_seed(0)
        return TransformerEncoder(
            20,
            output_size=32,
            attention_heads=2,
            linear_units=32,
            num_blocks=1,
            num_blocks_sd=1,
            num_inf=2,
            **kwargs,
        )

    # eval() so dropout does not make the two forward passes draw different
    # masks: the point of the comparison is the attention implementation.
    sdpa_encoder = build(use_sdpa=True).eval()
    attentions = [m for m in sdpa_encoder.modules() if hasattr(m, "use_sdpa")]
    assert attentions and all(m.use_sdpa for m in attentions)

    with torch.no_grad():
        default_out, _, _ = build().eval()(x, x_lens)
        sdpa_out, _, _ = sdpa_encoder(x, x_lens)
    assert default_out.shape == sdpa_out.shape
    assert torch.allclose(default_out, sdpa_out, atol=1e-5)

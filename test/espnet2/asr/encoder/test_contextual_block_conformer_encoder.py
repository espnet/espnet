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


@pytest.mark.parametrize(
    "block_size, hop_size, look_ahead",
    [
        (4, 2, 1),  # ten frames against a block of four: block processing
        (16, 8, 4),  # shorter than a block: the usual encoder
    ],
)
def test_use_sdpa_reaches_the_attention_and_matches_the_default(
    block_size, hop_size, look_ahead
):
    """`use_sdpa` has to be forwarded, and must not change the output.

    Block processing builds its own mask instead of using make_pad_mask's, and
    scaled_dot_product_attention adds a float mask to the scores rather than
    masking with it, so the block case also checks that this mask is bool.
    """
    x = torch.randn(2, 10, 20)
    x_lens = torch.LongTensor([10, 8])

    def build(**kwargs):
        torch.manual_seed(0)
        return ContextualBlockConformerEncoder(
            20,
            output_size=16,
            attention_heads=2,
            linear_units=16,
            num_blocks=2,
            input_layer="linear",
            cnn_module_kernel=3,
            block_size=block_size,
            hop_size=hop_size,
            look_ahead=look_ahead,
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
    assert torch.allclose(default_out, sdpa_out, atol=1e-5)


def test_use_sdpa_matches_the_default_when_streaming():
    """forward_infer builds the same block mask while decoding chunk by chunk."""
    x = torch.randn(1, 24, 20)

    def stream(**kwargs):
        torch.manual_seed(0)
        encoder = ContextualBlockConformerEncoder(
            20,
            output_size=16,
            attention_heads=2,
            linear_units=16,
            num_blocks=2,
            input_layer="linear",
            cnn_module_kernel=3,
            block_size=4,
            hop_size=2,
            look_ahead=1,
            **kwargs,
        ).eval()
        outputs, states = [], None
        with torch.no_grad():
            for start, is_final in ((0, False), (8, False), (16, True)):
                y, _, states = encoder(
                    x[:, start : start + 8],
                    torch.LongTensor([8]),
                    states,
                    is_final=is_final,
                    infer_mode=True,
                )
                outputs.append(y)
        return torch.cat(outputs, dim=1)

    assert torch.allclose(stream(), stream(use_sdpa=True), atol=1e-5)

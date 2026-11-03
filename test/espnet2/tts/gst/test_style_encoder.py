import pytest
import torch

from espnet2.tts.gst.style_encoder import StyleEncoder, StyleTokenLayer


def _build_style_encoder(**kwargs):
    torch.manual_seed(0)
    return StyleEncoder(
        idim=20,
        gst_tokens=4,
        gst_token_dim=16,
        gst_heads=2,
        conv_layers=2,
        conv_chans_list=(4, 8),
        gru_units=8,
        **kwargs,
    )


def test_style_encoder_forward_shape():
    encoder = _build_style_encoder().eval()
    style_embs = encoder(torch.randn(2, 30, 20))
    assert style_embs.shape == (2, 16)


def test_use_sdpa_reaches_the_attention_and_matches_the_default():
    speech = torch.randn(2, 30, 20)

    # eval() so dropout does not make the two forward passes draw different
    # masks: the point of the comparison is the attention implementation.
    sdpa_encoder = _build_style_encoder(use_sdpa=True).eval()
    attentions = [m for m in sdpa_encoder.modules() if hasattr(m, "use_sdpa")]
    assert attentions and all(m.use_sdpa for m in attentions)

    with torch.no_grad():
        default_out = _build_style_encoder().eval()(speech)
        sdpa_out = sdpa_encoder(speech)

    torch.testing.assert_close(sdpa_out, default_out, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("use_sdpa", [False, True])
def test_style_token_layer_trains_with_dropout(use_sdpa):
    layer = StyleTokenLayer(
        ref_embed_dim=8,
        gst_tokens=4,
        gst_token_dim=16,
        gst_heads=2,
        dropout_rate=0.1,
        use_sdpa=use_sdpa,
    ).train()
    out = layer(torch.randn(3, 8))
    assert out.shape == (3, 16)
    out.sum().backward()
    assert layer.gst_embs.grad is not None

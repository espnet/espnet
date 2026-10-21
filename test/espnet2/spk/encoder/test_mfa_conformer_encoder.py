import pytest
import torch

from espnet2.spk.encoder.conformer_encoder import MfaConformerEncoder


def _encoder(rel_pos_type, pos_enc_layer_type, selfattention_layer_type, **kwargs):
    torch.manual_seed(0)
    return MfaConformerEncoder(
        16,
        output_size=8,
        attention_heads=2,
        linear_units=16,
        num_blocks=2,
        rel_pos_type=rel_pos_type,
        pos_enc_layer_type=pos_enc_layer_type,
        selfattention_layer_type=selfattention_layer_type,
        cnn_module_kernel=3,
        **kwargs,
    )


@pytest.mark.parametrize(
    "rel_pos_type, pos_enc_layer_type, selfattention_layer_type",
    [
        ("legacy", "abs_pos", "selfattn"),
        ("latest", "rel_pos", "rel_selfattn"),
        ("legacy", "rel_pos", "rel_selfattn"),
    ],
)
def test_use_sdpa_reaches_every_attention(
    rel_pos_type, pos_enc_layer_type, selfattention_layer_type
):
    encoder = _encoder(
        rel_pos_type, pos_enc_layer_type, selfattention_layer_type, use_sdpa=True
    )
    attentions = [m for m in encoder.modules() if hasattr(m, "use_sdpa")]
    assert len(attentions) == 2 and all(m.use_sdpa for m in attentions)


@pytest.mark.parametrize("rel_pos_type", ["latest", "legacy"])
def test_use_sdpa_matches_the_default(rel_pos_type):
    x = torch.randn(2, 20, 16)

    # eval() so dropout does not make the two passes draw different masks.
    default_encoder = _encoder(rel_pos_type, "rel_pos", "rel_selfattn").eval()
    sdpa_encoder = _encoder(
        rel_pos_type, "rel_pos", "rel_selfattn", use_sdpa=True
    ).eval()
    with torch.no_grad():
        default_out = default_encoder(x)
        sdpa_out = sdpa_encoder(x)
    assert default_out.shape == sdpa_out.shape == (2, 16, 7)
    assert torch.allclose(default_out, sdpa_out, atol=1e-5)

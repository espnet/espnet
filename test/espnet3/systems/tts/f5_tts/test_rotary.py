"""Rotary positional embeddings.

A model forward only ever calls ``forward_from_seq_len`` on the default
configuration, so the xpos scale path, interpolation and partial rotation are
driven directly here.
"""

import pytest
import torch

from espnet3.systems.tts.f5_tts.rotary import (
    RotaryEmbedding,
    apply_rotary_pos_emb,
    rotate_half,
)


def test_rotate_half_swaps_and_negates_the_pairs():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    torch.testing.assert_close(rotate_half(x), torch.tensor([[-2.0, 1.0, -4.0, 3.0]]))


def test_frequencies_cover_the_full_rotary_dimension():
    rope = RotaryEmbedding(dim=8)

    freqs, scale = rope.forward_from_seq_len(5)

    assert freqs.shape == (1, 5, 8)
    assert scale == 1.0  # no xpos, so the scale is a plain scalar


def test_the_first_position_has_zero_rotation():
    rope = RotaryEmbedding(dim=8)

    freqs, _ = rope.forward_from_seq_len(3)

    torch.testing.assert_close(freqs[0, 0], torch.zeros(8))


def test_use_xpos_returns_a_per_position_scale_tensor():
    """xpos damps long-range attention; unused by F5 but kept for parity."""
    rope = RotaryEmbedding(dim=8, use_xpos=True)

    freqs, scale = rope.forward_from_seq_len(6)

    assert torch.is_tensor(scale)
    assert scale.shape == freqs.shape


def test_the_interpolation_factor_stretches_the_positions():
    """Position interpolation extends context by shrinking the angles."""
    plain, _ = RotaryEmbedding(dim=8).forward_from_seq_len(4)
    rope = RotaryEmbedding(dim=8, interpolation_factor=2.0)
    stretched, _ = rope.forward_from_seq_len(4)

    torch.testing.assert_close(stretched, plain / 2.0)


def test_an_interpolation_factor_below_one_is_rejected():
    with pytest.raises(AssertionError):
        RotaryEmbedding(dim=8, interpolation_factor=0.5)


def test_a_tensor_scale_is_sliced_to_the_sequence_length():
    """The xpos path passes a scale tensor, which must track the same window."""
    rope = RotaryEmbedding(dim=8, use_xpos=True)
    freqs, scale = rope.forward_from_seq_len(6)
    t = torch.randn(1, 2, 4, 8)  # b h n d, shorter than the 6-position grid

    out = apply_rotary_pos_emb(t, freqs, scale)

    assert out.shape == t.shape
    assert torch.isfinite(out).all()


def test_rotation_leaves_the_non_rotary_tail_untouched():
    """Partial rotary (GPT-J style): only the leading dims are rotated."""
    rope = RotaryEmbedding(dim=4)
    freqs, scale = rope.forward_from_seq_len(3)
    t = torch.randn(1, 3, 8)  # rotary dim 4, so the last 4 pass through

    out = apply_rotary_pos_emb(t, freqs, scale)

    torch.testing.assert_close(out[..., 4:], t[..., 4:])


def test_the_output_dtype_survives_the_fp32_computation():
    """Angles are computed in fp32, but the tensor keeps its own dtype."""
    rope = RotaryEmbedding(dim=8)
    freqs, scale = rope.forward_from_seq_len(3)
    t = torch.randn(1, 3, 8, dtype=torch.bfloat16)

    assert apply_rotary_pos_emb(t, freqs, scale).dtype == torch.bfloat16

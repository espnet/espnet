"""Tensor and tokenization helpers carried over verbatim from upstream F5-TTS.

These are pure functions, so they are pinned by behaviour here rather than only
being reached indirectly through a model forward.
"""

import torch

from espnet3.systems.tts.f5_tts.utils import (
    get_epss_timesteps,
    list_str_to_idx,
    list_str_to_tensor,
    maybe_masked_mean,
)


def test_masked_mean_ignores_padding():
    t = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [99.0, 99.0]]])
    mask = torch.tensor([[True, True, False]])

    torch.testing.assert_close(maybe_masked_mean(t, mask), torch.tensor([[2.0, 2.0]]))


def test_masked_mean_without_a_mask_averages_everything():
    t = torch.tensor([[[1.0], [3.0]]])

    torch.testing.assert_close(maybe_masked_mean(t, None), torch.tensor([[2.0]]))


def test_masked_mean_of_an_all_padding_row_is_zero_not_nan():
    """The denominator is clamped, so an empty row cannot divide by zero."""
    t = torch.tensor([[[5.0], [5.0]]])
    mask = torch.tensor([[False, False]])

    out = maybe_masked_mean(t, mask)

    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, torch.tensor([[0.0]]))


def test_utf8_tokenizer_pads_to_the_longest_string():
    out = list_str_to_tensor(["ab", "a"])

    assert out.shape == (2, 2)
    torch.testing.assert_close(out[0], torch.tensor([ord("a"), ord("b")]))
    assert out[1, 1] == -1  # padding


def test_char_tokenizer_maps_out_of_vocabulary_characters_to_zero():
    out = list_str_to_idx(["ab", "az"], {"a": 1, "b": 2})

    torch.testing.assert_close(out[0], torch.tensor([1, 2]))
    assert out[1, 1] == 0  # 'z' is not in the vocab


def test_char_tokenizer_pads_ragged_input():
    out = list_str_to_idx(["ab", "a"], {"a": 1, "b": 2}, padding_value=-1)

    assert out.shape == (2, 2)
    assert out[1, 1] == -1


def test_epss_uses_the_pruned_schedule_for_a_known_step_count():
    """The empirically pruned grids are non-uniform, unlike a linspace."""
    t = get_epss_timesteps(7, device="cpu", dtype=torch.float32)

    assert t.shape == (8,)  # 7 steps -> 8 grid points
    torch.testing.assert_close(t[0], torch.tensor(0.0))
    torch.testing.assert_close(t[-1], torch.tensor(1.0))
    steps = torch.diff(t)
    assert not torch.allclose(steps, steps[0].expand_as(steps))


def test_epss_falls_back_to_a_uniform_grid_for_other_step_counts():
    t = get_epss_timesteps(4, device="cpu", dtype=torch.float32)

    torch.testing.assert_close(t, torch.linspace(0, 1, 5))

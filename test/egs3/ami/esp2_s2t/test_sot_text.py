"""Tests for the SOT text builder in egs3/ami/esp2_s2t/dataset/sot_text.py.

These tests use lightweight supervision stubs rather than lhotse objects, so they stay
inside the suite's 2 s per-test budget and need no corpus.
"""

import importlib.util
from dataclasses import dataclass
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[4]
_RECIPE = _REPO / "egs3" / "ami" / "esp2_s2t"


def _load():
    spec = importlib.util.spec_from_file_location(
        "ami_s2t_sot_text", _RECIPE / "dataset" / "sot_text.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


st = _load()


@dataclass
class Sup:
    """Stands in for a lhotse SupervisionSegment."""

    speaker: str
    start: float
    duration: float
    text: str


def test_single_speaker_carries_no_separator():
    sups = [Sup("A", 0.0, 1.0, "hello")]
    out = st.build_sot_text(sups, separator="????")
    assert out == "<|0.00|> hello<|1.00|> <|endoftext|>"


def test_start_time_ordering_puts_the_earliest_onset_first():
    sups = [Sup("B", 1.0, 1.0, "second"), Sup("A", 0.0, 1.0, "first")]
    out = st.build_sot_text(sups, ordering="start_time", separator="????")
    assert out.startswith("<|0.00|> first<|1.00|> ???? <|1.00|> second<|2.00|>")


def test_longest_first_ordering_ranks_by_character_length_with_markup():
    # "B" starts later but its rendered block is longer, so it leads.
    sups = [Sup("A", 0.0, 1.0, "hi"), Sup("B", 1.0, 1.0, "a much longer turn")]
    out = st.build_sot_text(sups, ordering="longest_first", separator="????")
    assert out.startswith("<|1.00|> a much longer turn<|2.00|> ???? <|0.00|> hi")


def test_longest_first_is_stable_so_ties_keep_alphabetical_speaker_order():
    # Identical rendered lengths: the incoming (alphabetical) order survives.
    sups = [Sup("B", 0.0, 1.0, "xx"), Sup("A", 0.0, 1.0, "yy")]
    out = st.build_sot_text(sups, ordering="longest_first", separator="????")
    assert out == "<|0.00|> yy<|1.00|> ???? <|0.00|> xx<|1.00|> <|endoftext|>"


def test_lowercase_can_be_turned_off():
    sups = [Sup("A", 0.0, 1.0, "Hello There")]
    assert "Hello There" in st.build_sot_text(sups, lowercase=False)
    assert "hello there" in st.build_sot_text(sups, lowercase=True)


def test_a_gap_within_the_pause_merges_into_one_block():
    sups = [Sup("A", 0.0, 1.0, "one"), Sup("A", 2.5, 1.0, "two")]
    out = st.build_sot_text(sups, max_timestamp_pause=2.0)
    assert out == "<|0.00|> one two<|3.50|> <|endoftext|>"


def test_a_gap_beyond_the_pause_splits_into_two_timestamped_runs():
    sups = [Sup("A", 0.0, 1.0, "one"), Sup("A", 4.0, 1.0, "two")]
    out = st.build_sot_text(sups, max_timestamp_pause=2.0)
    assert out == "<|0.00|> one<|1.00|><|4.00|> two<|5.00|> <|endoftext|>"


def test_a_nested_supervision_keeps_the_outer_end_time():
    # The second segment sits inside the first; merging must not truncate.
    sups = [Sup("A", 0.0, 5.0, "long"), Sup("A", 1.0, 1.0, "short")]
    out = st.build_sot_text(sups, max_timestamp_pause=2.0)
    assert out == "<|0.00|> long short<|5.00|> <|endoftext|>"


def test_empty_text_supervisions_are_dropped():
    sups = [Sup("A", 0.0, 1.0, "  "), Sup("B", 1.0, 1.0, "real")]
    out = st.build_sot_text(sups, separator="????")
    assert "????" not in out
    assert out == "<|1.00|> real<|2.00|> <|endoftext|>"


def test_separator_is_configurable():
    sups = [Sup("A", 0.0, 1.0, "a"), Sup("B", 1.0, 1.0, "b")]
    assert " <sc> " in st.build_sot_text(sups, separator="<sc>")


def test_timestamps_can_be_disabled():
    sups = [Sup("A", 0.0, 1.0, "a"), Sup("B", 1.0, 1.0, "b")]
    out = st.build_sot_text(sups, use_timestamps=False, separator="????")
    assert out == "a ???? b <|endoftext|>"


def test_the_end_of_sequence_token_is_configurable_and_omittable():
    sups = [Sup("A", 0.0, 1.0, "a")]
    assert st.build_sot_text(sups, eos=None) == "<|0.00|> a<|1.00|>"


def test_an_unknown_ordering_is_rejected_by_name():
    sups = [Sup("A", 0.0, 1.0, "a")]
    with pytest.raises(ValueError, match="loudest_first"):
        st.build_sot_text(sups, ordering="loudest_first")


def test_timestamps_snap_to_the_20_ms_grid():
    sups = [Sup("A", 0.123, 0.5, "a")]
    out = st.build_sot_text(sups)
    assert out == "<|0.12|> a<|0.62|> <|endoftext|>"


def test_no_supervisions_yields_only_the_end_token():
    assert st.build_sot_text([]) == "<|endoftext|>"


def test_prompt_is_prepended_verbatim_with_no_separating_space():
    """S2T text begins <language><task> with the first timestamp flush after."""
    sups = [Sup("A", 0.0, 1.0, "hello")]
    out = st.build_sot_text(sups, prompt="<|en|><|transcribe|>", eos=None)
    assert out == "<|en|><|transcribe|><|0.00|> hello<|1.00|>"


def test_prompt_defaults_to_absent_so_existing_callers_are_unchanged():
    sups = [Sup("A", 0.0, 1.0, "hello")]
    assert st.build_sot_text(sups, eos=None) == "<|0.00|> hello<|1.00|>"


def test_a_training_line_carries_no_trailing_end_token():
    """_calc_att_loss appends eos itself; a text ending in one trains on two."""
    sups = [Sup("A", 0.0, 1.0, "hello")]
    out = st.build_sot_text(sups, prompt="<|en|><|transcribe|>", eos=None)
    assert not out.endswith("<|endoftext|>")


def test_prompt_survives_a_multi_speaker_line():
    sups = [Sup("A", 0.0, 1.0, "a"), Sup("B", 1.0, 1.0, "b")]
    out = st.build_sot_text(
        sups, prompt="<|en|><|transcribe|>", separator="????", eos=None
    )
    assert out == "<|en|><|transcribe|><|0.00|> a<|1.00|> ???? <|1.00|> b<|2.00|>"


def test_an_empty_line_is_still_only_the_end_token_even_with_a_prompt():
    """No content means no prompt either; an empty target must stay empty."""
    assert st.build_sot_text([], prompt="<|en|><|transcribe|>") == "<|endoftext|>"


# ---------------------------------------------------------------------------
# text_norm: applied to a segment's words, never to the markup
# ---------------------------------------------------------------------------


def test_text_norm_runs_before_the_timestamps_wrap_the_text():
    """The normalizer must see words only.

    A segment is normalized and then wrapped, so neither the timestamps nor
    the separator can reach the normalizer. One that swallowed either would
    produce a line the metrics cannot split.
    """
    seen = []

    def norm(text):
        seen.append(text)
        return text.replace("dont", "do not")

    sups = [Sup("A", 0.0, 1.0, "i dont know"), Sup("B", 2.0, 1.0, "me neither")]
    out = st.build_sot_text(
        sups, ordering="start_time", separator="????", text_norm=norm
    )

    assert seen == ["i dont know", "me neither"]  # no markup, no separator
    assert "<|0.00|> i do not know<|1.00|> ???? <|2.00|> me neither<|3.00|>" in out


def test_text_norm_replaces_lowercasing_rather_than_stacking_with_it():
    """A normalizer case-folds on its own, so lowercase must not run too.

    The normalizer is given the raw supervision text. Case-folding first
    would hand it a different input than the targets were written from.
    """
    seen = []

    def norm(text):
        seen.append(text)
        return text.lower()

    st.build_sot_text(
        [Sup("A", 0.0, 1.0, "Hello There")], lowercase=True, text_norm=norm
    )
    assert seen == ["Hello There"]


def test_a_segment_that_normalizes_to_nothing_is_dropped():
    """Emptiness is judged after normalizing, not before.

    A segment holding only symbols survives the strip() but normalizes away.
    Keeping it would leave a bare timestamp pair with no words between them,
    which the DER segment parser would read as a real speech segment.
    """
    # 4 s apart, so merge_supervisions keeps them as two segments.
    sups = [Sup("A", 0.0, 1.0, "%%%"), Sup("A", 5.0, 1.0, "real words")]
    out = st.build_sot_text(
        sups, separator="????", text_norm=lambda t: "" if t == "%%%" else t
    )
    assert "<|0.00|>" not in out
    assert out.startswith("<|5.00|> real words<|6.00|>")


def test_a_speaker_whose_every_segment_normalizes_away_adds_no_separator():
    sups = [Sup("A", 0.0, 1.0, "%%%"), Sup("B", 2.0, 1.0, "real words")]
    out = st.build_sot_text(
        sups, separator="????", text_norm=lambda t: "" if t == "%%%" else t
    )
    assert "????" not in out

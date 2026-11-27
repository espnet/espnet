"""Tests for the SOT text builder in espnet3/systems/esp2_s2t/src/sot/sot_text.py.

These tests use small supervision stubs and no lhotse objects. So each test stays
inside the 2 s per-test time limit and needs no corpus.
"""

from dataclasses import dataclass

import pytest

import espnet3.systems.esp2_s2t.src.sot.sot_text as st  # noqa: E402


@dataclass
class Sup:
    """A stub for a lhotse SupervisionSegment."""

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
    # "B" starts later, but its rendered block is longer, so it comes first.
    sups = [Sup("A", 0.0, 1.0, "hi"), Sup("B", 1.0, 1.0, "a much longer turn")]
    out = st.build_sot_text(sups, ordering="longest_first", separator="????")
    assert out.startswith("<|1.00|> a much longer turn<|2.00|> ???? <|0.00|> hi")


def test_longest_first_is_stable_so_ties_keep_alphabetical_speaker_order():
    # The rendered lengths are equal, so the input (alphabetical) order is kept.
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
    # The second segment is inside the first, so merging must keep the outer end.
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
    """The prompt is added unchanged at the start, with no space after it.

    S2T text begins with <language><task>, and the first timestamp follows directly.
    """
    sups = [Sup("A", 0.0, 1.0, "hello")]
    out = st.build_sot_text(sups, prompt="<|en|><|transcribe|>", eos=None)
    assert out == "<|en|><|transcribe|><|0.00|> hello<|1.00|>"


def test_prompt_defaults_to_absent_so_existing_callers_are_unchanged():
    sups = [Sup("A", 0.0, 1.0, "hello")]
    assert st.build_sot_text(sups, eos=None) == "<|0.00|> hello<|1.00|>"


def test_a_training_line_carries_no_trailing_end_token():
    """With eos=None, the training text does not end with <|endoftext|>.

    _calc_att_loss appends eos itself. A text that already ends with eos would
    train on two of them.
    """
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
    """An empty line gives only the end token, even with a prompt.

    Without content there is no prompt either, so an empty target stays empty.
    """
    assert st.build_sot_text([], prompt="<|en|><|transcribe|>") == "<|endoftext|>"


# ---------------------------------------------------------------------------
# text_norm is applied to the words of a segment and never to the markup.
# ---------------------------------------------------------------------------


def test_text_norm_runs_before_the_timestamps_wrap_the_text():
    """text_norm gets only the words of each segment.

    A segment is normalized before the timestamps are added, so the normalizer
    never sees the timestamps or the separator. A normalizer that removed either
    of them would produce a line that the metrics cannot split.
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
    """With text_norm set, lowercase=True does not lowercase the text first.

    A normalizer does its own case-folding, so it gets the raw supervision text.
    Lowercasing first would give it a different input from the one that the
    targets were written from.
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
    """A segment that is empty after normalization is dropped.

    The emptiness check runs after normalization. strip() keeps a segment with only
    symbols, and normalization then removes all of it. Keeping it would leave a
    timestamp pair with no words between them, and the DER segment parser would
    read that pair as a real speech segment.
    """
    # The segments are 4 s apart, so merge_supervisions keeps them separate.
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

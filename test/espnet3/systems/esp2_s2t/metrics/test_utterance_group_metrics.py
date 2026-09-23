"""Tests for the S2T metrics in espnet3/systems/esp2_s2t/metrics/.

These pin the two properties a caller cannot see from the outside: that the
speaker-change symbol comes from the constructor rather than from a constant,
and that the reported keys say which scope the score has.
"""

import pytest

pytest.importorskip("scipy")
pytest.importorskip("editdistance")

from espnet3.systems.esp2_s2t.metrics.cpwer import (  # noqa: E402
    UtteranceGroupCpWER,
    split_speakers,
)
from espnet3.systems.esp2_s2t.metrics.der import (  # noqa: E402
    UtteranceGroupDER,
    segments_from_sot,
)


@pytest.mark.parametrize("symbol", ["????", "<sc>", "@@", "|SPK|"])
def test_split_speakers_uses_the_symbol_it_is_given(symbol):
    """A metric cannot know which symbol a checkpoint was trained with."""
    text = f"the cat sat {symbol} a dog barked"
    assert split_speakers(text, None, symbol) == ["the cat sat", "a dog barked"]


def test_split_speakers_does_not_split_on_a_symbol_it_was_not_given():
    """The wrong symbol must leave the text as one block, not guess."""
    assert split_speakers("the cat sat @@ a dog barked", None, "????") == [
        "the cat sat @@ a dog barked"
    ]


@pytest.mark.parametrize("symbol", ["????", "<sc>", "@@"])
def test_segments_from_sot_uses_the_symbol_it_is_given(symbol):
    text = f"<|0.00|> a<|1.20|> {symbol} <|2.00|> b<|3.50|>"
    assert segments_from_sot(text, symbol) == [(0, 0.0, 1.2), (1, 2.0, 3.5)]


def test_cpwer_reports_the_utterance_group_key(tmp_path):
    """The key carries the scope, because metrics.json is what gets quoted.

    A score labelled plainly ``cpWER`` invites comparison with session-level
    numbers, which are a harder problem: there one speaker assignment has to
    serve a whole meeting, while this one is free to choose per group.
    """
    ref, hyp = tmp_path / "ref.scp", tmp_path / "hyp.scp"
    ref.write_text("u1 the cat sat ???? a dog barked\n", encoding="utf-8")
    hyp.write_text("u1 a dog barked ???? the cat sat\n", encoding="utf-8")
    result = UtteranceGroupCpWER(clean_types=None)(
        {"ref": ref, "hyp": hyp}, "test", tmp_path
    )
    assert result == {"ug_cpWER": 0.0}


def test_der_reports_the_utterance_group_key(tmp_path):
    """Same reasoning as cpWER: every group is its own RTTM file."""
    pytest.importorskip("pytest")
    ref, hyp = tmp_path / "ref.scp", tmp_path / "hyp.scp"
    line = "u1 <|0.00|> a<|1.00|> ???? <|2.00|> b<|3.00|>\n"
    ref.write_text(line, encoding="utf-8")
    hyp.write_text(line, encoding="utf-8")
    metric = UtteranceGroupDER(clean_types=None)
    try:
        result = metric({"ref": ref, "hyp": hyp}, "test", tmp_path)
    except FileNotFoundError:
        pytest.skip("md-eval.pl is not built in this checkout")
    assert set(result) == {"ug_DER"}


def test_the_default_symbol_is_the_one_the_released_checkpoint_uses():
    """Four question marks, one Whisper BPE token."""
    assert UtteranceGroupCpWER().speaker_change_symbol == "????"
    assert UtteranceGroupDER().speaker_change_symbol == "????"

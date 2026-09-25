"""Tests for the S2T metrics in espnet3/systems/esp2_s2t/metrics/.

These pin the two properties a caller cannot see from the outside: that the
speaker-change symbol comes from the constructor rather than from a constant,
and that the reported keys say which scope the score has.
"""

import pytest

pytest.importorskip("meeteval")
pytest.importorskip("pyannote.metrics")

from espnet3.systems.esp2_s2t.metrics.cpwer import (  # noqa: E402
    UtteranceGroupCpWER,
    split_speakers,
)
from espnet3.systems.esp2_s2t.metrics.der import (  # noqa: E402
    UtteranceGroupDER,
    parse_sot_segments,
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
    assert parse_sot_segments(text, symbol) == [(0, 0.0, 1.2), (1, 2.0, 3.5)]


def test_cpwer_reports_the_utterance_group_key(tmp_path):
    """The key carries the scope, because metrics.json is what gets quoted.

    A score labelled plainly ``cpWER`` invites comparison with session-level
    numbers, which are a harder problem: there one speaker assignment has to
    serve a whole meeting, while this one is free to choose per group.
    """
    ref, hyp = tmp_path / "ref.scp", tmp_path / "hyp.scp"
    ref.write_text("u1 the cat sat ???? a dog barked\n", encoding="utf-8")
    hyp.write_text("u1 a dog barked ???? the cat sat\n", encoding="utf-8")
    result = UtteranceGroupCpWER("????", clean_types=None)(
        {"ref": ref, "hyp": hyp}, "test", tmp_path
    )
    assert result == {"ug_cpWER": 0.0}


def test_der_reports_the_utterance_group_key(tmp_path):
    """Same reasoning as cpWER: every group is scored as its own recording."""
    ref, hyp = tmp_path / "ref.scp", tmp_path / "hyp.scp"
    line = "u1 <|0.00|> a<|1.00|> ???? <|2.00|> b<|3.00|>\n"
    ref.write_text(line, encoding="utf-8")
    hyp.write_text(line, encoding="utf-8")
    metric = UtteranceGroupDER("????", clean_types=None)
    result = metric({"ref": ref, "hyp": hyp}, "test", tmp_path)
    assert set(result) == {"ug_DER"}


def test_the_symbol_has_no_default():
    """A wrong symbol scores badly rather than failing, so it must be given.

    Left to a default, a checkpoint that separates speakers some other way
    would score every group as one block and report a plausible number.
    """
    with pytest.raises(TypeError, match="speaker_change_symbol"):
        UtteranceGroupCpWER()
    with pytest.raises(TypeError, match="speaker_change_symbol"):
        UtteranceGroupDER()


@pytest.mark.parametrize("metric_cls", [UtteranceGroupCpWER, UtteranceGroupDER])
@pytest.mark.parametrize("symbol", [None, ""])
def test_a_metric_refuses_a_symbol_it_cannot_split_on(metric_cls, symbol):
    """Refusing at construction is the only place this can be caught.

    Neither class has a default symbol, but a config that names the key and
    leaves it empty passes None through. Splitting on nothing scores a whole
    utterance group as one speaker, which reads as a poor result rather than as
    a misconfiguration.
    """
    with pytest.raises(ValueError, match="speaker_change_symbol"):
        metric_cls(symbol)


@pytest.mark.parametrize(
    "case, reference, hypothesis, expected",
    [
        # One speaker over 0-2 s, heard only to 1 s: 1 s missed of 2 s.
        (
            "half the reference speech missed",
            "<|0.00|> a<|2.00|>",
            "<|0.00|> a<|1.00|>",
            50.0,
        ),
        # Two speakers a second each, heard as one speaker over the whole
        # window: the second second is attributed to the wrong speaker.
        (
            "second speaker heard as the first",
            "<|0.00|> a<|1.00|> ???? <|1.00|> b<|2.00|>",
            "<|0.00|> a<|2.00|>",
            50.0,
        ),
        ("nothing hypothesised", "<|0.00|> a<|2.00|>", "", 100.0),
        ("identical", "<|0.00|> a<|2.00|>", "<|0.00|> a<|2.00|>", 0.0),
    ],
)
def test_der_counts_the_seconds_it_should(
    case, reference, hypothesis, expected, tmp_path
):
    """The scoring arithmetic, on cases small enough to work out by hand.

    Every value here is error seconds over reference seconds with no collar:
    the reference is two seconds of speech, and each case misplaces a known
    number of them. Without this the only unconditional DER assertion was a
    0.0, which an accumulation that scored nothing at all also produces.
    """
    ref, hyp = tmp_path / "ref.scp", tmp_path / "hyp.scp"
    ref.write_text(f"u1 {reference}\n", encoding="utf-8")
    hyp.write_text(f"u1 {hypothesis}\n", encoding="utf-8")

    metric = UtteranceGroupDER("????", collar=0.0)
    assert metric({"ref": ref, "hyp": hyp}, "test", tmp_path)["ug_DER"] == expected

"""Tests for the S2T metrics in espnet3/systems/esp2_s2t/metrics/.

These tests check two properties that a caller cannot see from the outside. The
metrics take the speaker-change symbol from the constructor and have no constant
for it. The reported keys show the scope of the score.
"""

import json

import pytest

pytest.importorskip("meeteval")
pytest.importorskip("pyannote.metrics")

from espnet3.systems.esp2_s2t.metrics.cpwer import (  # noqa: E402
    UtteranceGroupCpWER,
    split_speakers,
)
from espnet3.systems.esp2_s2t.metrics.der import (  # noqa: E402
    UtteranceGroupDER,
    _parse_sot_segments,
)


@pytest.mark.parametrize("symbol", ["????", "<sc>", "@@", "|SPK|"])
def test_split_speakers_uses_the_symbol_it_is_given(symbol):
    """split_speakers splits the text on the symbol passed to it.

    A metric cannot know which symbol a checkpoint was trained with.
    """
    text = f"the cat sat {symbol} a dog barked"
    assert split_speakers(text, None, symbol) == ["the cat sat", "a dog barked"]


def test_split_speakers_does_not_split_on_a_symbol_it_was_not_given():
    """With the wrong symbol, split_speakers keeps the text as one block.

    It does not try to guess another separator.
    """
    assert split_speakers("the cat sat @@ a dog barked", None, "????") == [
        "the cat sat @@ a dog barked"
    ]


@pytest.mark.parametrize("symbol", ["????", "<sc>", "@@"])
def test_segments_from_sot_uses_the_symbol_it_is_given(symbol):
    text = f"<|0.00|> a<|1.20|> {symbol} <|2.00|> b<|3.50|>"
    assert _parse_sot_segments(text, symbol) == [(0, 0.0, 1.2), (1, 2.0, 3.5)]


def test_cpwer_reports_the_utterance_group_key(tmp_path):
    """The cpWER metric reports its score under the key ug_cpWER.

    The key shows the scope, because people quote the numbers in metrics.json. A
    score labelled only ``cpWER`` could be compared with session-level numbers.
    That is a harder problem, because one speaker assignment must cover a whole
    meeting. This metric chooses the assignment for each group.
    """
    ref, hyp = tmp_path / "ref.scp", tmp_path / "hyp.scp"
    ref.write_text("u1 the cat sat ???? a dog barked\n", encoding="utf-8")
    hyp.write_text("u1 a dog barked ???? the cat sat\n", encoding="utf-8")
    result = UtteranceGroupCpWER("????", clean_types=None)(
        {"ref": ref, "hyp": hyp}, "test", tmp_path
    )
    assert result == {"ug_cpWER": 0.0}


def test_der_reports_the_utterance_group_key(tmp_path):
    """The DER metric reports its score under the key ug_DER.

    The reason is the same as for cpWER. Every group is scored as its own
    recording.
    """
    ref, hyp = tmp_path / "ref.scp", tmp_path / "hyp.scp"
    line = "u1 <|0.00|> a<|1.00|> ???? <|2.00|> b<|3.00|>\n"
    ref.write_text(line, encoding="utf-8")
    hyp.write_text(line, encoding="utf-8")
    metric = UtteranceGroupDER("????", clean_types=None)
    result = metric({"ref": ref, "hyp": hyp}, "test", tmp_path)
    assert set(result) == {"ug_DER"}


def test_the_symbol_has_no_default():
    """Neither UtteranceGroupCpWER nor UtteranceGroupDER has a default symbol.

    A wrong symbol gives a bad score and raises no error. With a default, a
    checkpoint that separates speakers in another way would score every group as
    one block, and the reported number would look plausible.
    """
    with pytest.raises(TypeError, match="speaker_change_symbol"):
        UtteranceGroupCpWER()
    with pytest.raises(TypeError, match="speaker_change_symbol"):
        UtteranceGroupDER()


@pytest.mark.parametrize("metric_cls", [UtteranceGroupCpWER, UtteranceGroupDER])
@pytest.mark.parametrize("symbol", [None, ""])
def test_a_metric_refuses_a_symbol_it_cannot_split_on(metric_cls, symbol):
    """A metric raises ValueError for a None or empty symbol.

    The constructor is the only place where this mistake can be caught. Neither
    class has a default symbol, but a config that has the key with no value passes
    None. Splitting on nothing scores a whole utterance group as one speaker. The
    result looks like a poor score and does not show the config error.
    """
    with pytest.raises(ValueError, match="speaker_change_symbol"):
        metric_cls(symbol)


@pytest.mark.parametrize(
    "case, reference, hypothesis, expected",
    [
        # One speaker from 0 to 2 s is heard only until 1 s, so 1 s of 2 s is missed.
        (
            "half the reference speech missed",
            "<|0.00|> a<|2.00|>",
            "<|0.00|> a<|1.00|>",
            50.0,
        ),
        # Two speakers speak for one second each and are heard as one speaker for
        # the whole window. So the second second is given to the wrong speaker.
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
    """The DER metric gives the right values on small cases checked by hand.

    Every value is error seconds divided by reference seconds, with no collar. The
    reference has two seconds of speech, and each case misplaces a known number of
    them. Without this test, the only unconditional DER assertion was a 0.0, and an
    accumulation that scored nothing would also give 0.0.
    """
    ref, hyp = tmp_path / "ref.scp", tmp_path / "hyp.scp"
    ref.write_text(f"u1 {reference}\n", encoding="utf-8")
    hyp.write_text(f"u1 {hypothesis}\n", encoding="utf-8")

    metric = UtteranceGroupDER("????", collar=0.0)
    assert metric({"ref": ref, "hyp": hyp}, "test", tmp_path)["ug_DER"] == expected


_GROUPS = [
    (
        "<|0.00|> the cat sat<|1.00|> ???? <|1.00|> a dog<|2.00|>",
        "<|0.00|> a dog<|1.50|> ???? <|0.50|> the bat sat<|1.00|>",
    ),
    ("<|0.00|> yes<|0.80|>", "<|0.00|> yes no<|1.20|>"),
    ("<|0.00|> one two three<|2.50|>", ""),
]


def _all_metrics():
    from espnet3.systems.esp2_s2t.metrics.speaker_count import (
        UtteranceGroupSpeakerCountAccuracy,
    )
    from espnet3.systems.esp2_s2t.metrics.wer import UtteranceGroupWER

    return [
        lambda: UtteranceGroupCpWER("????"),
        lambda: UtteranceGroupDER("????", collar=0.0),
        lambda: UtteranceGroupSpeakerCountAccuracy("????"),
        lambda: UtteranceGroupWER("????"),
    ]


@pytest.mark.parametrize(
    "make", _all_metrics(), ids=["cpWER", "DER", "speaker_count", "WER"]
)
def test_one_batch_chunks_and_merged_ranks_agree(make):
    """One batch, single-group chunks and merged ranks give the same score.

    The score sums counts, so the way that the groups are split cannot change it. A
    metric that averaged the per-group rates would pass a one-group test and fail
    here, where the groups have different lengths.
    """
    refs = [ref for ref, _ in _GROUPS]
    hyps = [hyp for _, hyp in _GROUPS]
    whole = make()
    whole.update(ref=refs, hyp=hyps)
    chunked = make()
    ranks = [make() for _ in _GROUPS]
    for rank, (ref, hyp) in zip(ranks, _GROUPS):
        chunked.update(ref=[ref], hyp=[hyp])
        rank.update(ref=[ref], hyp=[hyp])
    merged = make()
    merged.load_state(merged.merge_states([rank.state() for rank in ranks]))

    assert whole.compute() == chunked.compute() == merged.compute()
    whole.reset()
    with pytest.raises(ValueError):
        whole.compute()


def test_the_cpwer_report_is_keyed_by_utterance_id(tmp_path):
    """cpwer_per_utt.json uses the utterance ids as keys, in input order.

    update() never gets the ids, so the report must recover them in order.
    """
    ref, hyp = tmp_path / "ref.scp", tmp_path / "hyp.scp"
    ref.write_text(
        "".join(f"g{i} {r}\n" for i, (r, _) in enumerate(_GROUPS)), encoding="utf-8"
    )
    hyp.write_text(
        "".join(f"g{i} {h}\n" for i, (_, h) in enumerate(_GROUPS)), encoding="utf-8"
    )
    metric = UtteranceGroupCpWER("????")
    metric.chunk_size = 2
    metric({"ref": ref, "hyp": hyp}, "test", tmp_path)

    per_utt = json.loads((tmp_path / "test" / "cpwer_per_utt.json").read_text())
    assert list(per_utt) == ["g0", "g1", "g2"]
    assert per_utt["g2"]["errors"] == 3


def test_speaker_counting_counts_cleaned_blocks():
    """A block that is empty after cleaning does not count as a speaker.

    ESPnet2 counts speakers in the same way.
    """
    from espnet3.systems.esp2_s2t.metrics.speaker_count import (
        UtteranceGroupSpeakerCountAccuracy,
    )

    metric = UtteranceGroupSpeakerCountAccuracy("????")
    metric.update(
        ref=["<|0.00|> a<|1.00|> ???? <|1.00|> b<|2.00|>", "<|0.00|> c<|1.00|>"],
        hyp=["<|0.00|> a b<|2.00|>", "<|0.00|> c<|1.00|> ???? <|1.00|><|2.00|>"],
    )
    assert metric.compute() == {"ug_spk_count_acc": 50.0}


def test_wer_puts_the_reference_in_fifo_order_and_keeps_the_hypothesis_order():
    """WER sorts the reference speakers by start time.

    The hypothesis keeps the speaker order that the model produced. A reference
    that lists a later speaker first gives no error for a hypothesis in time
    order, and a hypothesis in the reference order is counted as wrong.
    """
    from espnet3.systems.esp2_s2t.metrics.wer import UtteranceGroupWER

    longest_first = "<|1.00|> b c d<|2.00|> ???? <|0.00|> a<|1.00|>"
    in_order = "<|0.00|> a<|1.00|> ???? <|1.00|> b c d<|2.00|>"
    swapped = longest_first

    metric = UtteranceGroupWER("????")
    metric.update(ref=[longest_first], hyp=[in_order])
    assert metric.compute() == {"ug_WER": 0.0}
    metric.reset()
    metric.update(ref=[longest_first], hyp=[swapped])
    assert metric.compute()["ug_WER"] > 0

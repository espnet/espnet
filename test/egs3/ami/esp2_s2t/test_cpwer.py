"""Recipe-level tests for the utterance-group cpWER metric.

The scorer itself lives in espnet3/systems/esp2_s2t/metrics/cpwer.py;
what these pin is the score it gives on this recipe's corpus.
"""

import json

import ami_sot_paths
import pytest

pytest.importorskip("scipy")
pytest.importorskip("editdistance")

from espnet3.systems.esp2_s2t.metrics import cpwer as cp  # noqa: E402

SEP = "????"


def test_edit_counts_counts_each_error_kind():
    cor, sub, dele, ins = cp.edit_counts(["a", "b", "c"], ["a", "x", "c", "d"])
    assert (cor, sub, dele, ins) == (2, 1, 0, 1)


def test_group_cpwer_finds_the_best_speaker_permutation():
    """Speaker blocks arrive in an arbitrary order, so the match is optimal."""
    ref = ["the cat sat", "a dog barked"]
    swapped = ["a dog barked", "the cat sat"]
    acc = cp.group_cpwer(ref, swapped)
    assert acc["sub"] == 0 and acc["del"] == 0 and acc["ins"] == 0
    assert acc["ref_len"] == 6


def test_group_cpwer_pads_a_missing_hypothesis_speaker_as_deletions():
    acc = cp.group_cpwer(["the cat sat", "a dog barked"], ["the cat sat"])
    assert acc["del"] == 3
    assert acc["ref_len"] == 6


def test_group_cpwer_counts_an_empty_reference_as_insertions():
    acc = cp.group_cpwer([], ["x y"])
    assert acc["ins"] == 2
    assert acc["ref_len"] == 0


def test_metric_scores_two_aligned_scp_files(tmp_path):
    ref = tmp_path / "ref.scp"
    hyp = tmp_path / "hyp.scp"
    sep = SEP
    ref.write_text(f"u1 the cat sat {sep} a dog barked\n", encoding="utf-8")
    hyp.write_text(f"u1 a dog barked {sep} the cat sat\n", encoding="utf-8")
    metric = cp.UtteranceGroupCpWER(SEP, clean_types=None)
    result = metric({"ref": ref, "hyp": hyp}, "test", tmp_path)
    assert result == {"ug_cpWER": 0.0}


def test_metric_writes_a_by_speaker_count_side_file(tmp_path):
    ref = tmp_path / "ref.scp"
    hyp = tmp_path / "hyp.scp"
    sep = SEP
    ref.write_text(f"u1 the cat sat {sep} a dog barked\n", encoding="utf-8")
    hyp.write_text(f"u1 the cat sat {sep} a dog barked\n", encoding="utf-8")
    cp.UtteranceGroupCpWER(SEP, clean_types=None)(
        {"ref": ref, "hyp": hyp}, "test", tmp_path
    )
    assert (tmp_path / "test" / "cpwer_by_num_speakers.json").is_file()


def test_metric_raises_when_every_reference_is_empty(tmp_path):
    """A run whose references all came out empty must fail loudly.

    total["ref_len"] would be 0, and pct() returns 0.0 for a zero denominator, so an
    unguarded aggregate would misreport a perfect 0.00% instead. The realistic way in is
    src/inference.py's ``data.get("text", "")``, which silently empties every reference
    if the dataset's text key is ever renamed.
    """
    ref = tmp_path / "ref.scp"
    hyp = tmp_path / "hyp.scp"
    ref.write_text("u1\nu2\n", encoding="utf-8")
    hyp.write_text("u1 some words\nu2 more words\n", encoding="utf-8")
    metric = cp.UtteranceGroupCpWER(SEP, clean_types=None)
    with pytest.raises(ValueError, match="empty"):
        metric({"ref": ref, "hyp": hyp}, "test", tmp_path)


def test_by_speaker_count_file_reports_null_not_zero_for_empty_references(
    tmp_path,
):
    """cpwer_by_num_speakers.json must agree with cpwer_per_utt.json.

    Both describe the same zero-reference-speaker groups, and cpwer_per_utt.json already
    reports ``null`` for one (via utt_cpwer). The by-speaker-count file must not
    independently call the unguarded ``pct`` and report a contradictory 0.0 for the same
    group.
    """
    ref = tmp_path / "ref.scp"
    hyp = tmp_path / "hyp.scp"
    ref.write_text("u1\nu2 the cat sat\n", encoding="utf-8")
    hyp.write_text("u1 spurious words\nu2 the cat sat\n", encoding="utf-8")
    cp.UtteranceGroupCpWER(SEP, clean_types=None)(
        {"ref": ref, "hyp": hyp}, "test", tmp_path
    )
    by_nspk = json.loads(
        (tmp_path / "test" / "cpwer_by_num_speakers.json").read_text(encoding="utf-8")
    )
    per_utt = json.loads(
        (tmp_path / "test" / "cpwer_per_utt.json").read_text(encoding="utf-8")
    )
    assert by_nspk["0"]["cpwer"] is None
    assert per_utt["u1"]["cpwer"] is None


# needs_reference_decode only checks that the decode directory is there; the
# cpWER regression reads the word-level "text" inside it.
_RECORDED_TEXT = (
    (ami_sot_paths.REFERENCE_DECODE / "text")
    if ami_sot_paths.REFERENCE_DECODE
    else None
)


@pytest.mark.skipif(
    _RECORDED_TEXT is None or not _RECORDED_TEXT.is_file(),
    reason="the recorded decode directory holds no 'text'",
)
@ami_sot_paths.needs_corpus
@ami_sot_paths.needs_reference_decode
@pytest.mark.execution_timeout(60.0)
def test_cpwer_reproduces_the_recorded_full_test_set_score(tmp_path):
    """The port must agree with the number the native run already produced."""
    ref = tmp_path / "ref.scp"
    hyp = tmp_path / "hyp.scp"

    # Both sides may predate this recipe and spell the separator "<sc>": the
    # recorded decode came from the older decode.py path, and a corpus
    # prepared before the recipe writes it that way too. The metric splits on
    # the checkpoint's own symbol, so convert here rather than teach it a
    # spelling this recipe never writes.
    def _as_configured(text):
        return text.replace("<sc>", SEP)

    ref.write_text(
        _as_configured(ami_sot_paths.TEST_TEXT.read_text(encoding="utf-8")),
        encoding="utf-8",
    )
    hyp.write_text(
        _as_configured(_RECORDED_TEXT.read_text(encoding="utf-8")), encoding="utf-8"
    )
    result = cp.UtteranceGroupCpWER(SEP, clean_types=["whisper_en"])(
        {"ref": ref, "hyp": hyp}, "test", tmp_path
    )
    assert result["ug_cpWER"] == 27.65


def test_split_speakers_follows_the_symbol_it_is_given():
    """split_speakers must use its argument, not a literal.

    Without it, "@@" is not recognized as a separator and the whole text
    stays one block.
    """
    assert cp.split_speakers("a b @@ c d", None, "@@") == ["a b", "c d"]

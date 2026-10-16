"""Tests for the phone recognition metrics.

# Test Case Summary
| Test | Description |
|---|---|
| test_clean_ipa_text_normalizes | Spaces, punctuation and Latin g are normalized away |
| test_clean_ipa_text_strips_angle_brackets | Special tokens must go first |
| test_segment_ipa_keeps_diacritics_together | Multi-codepoint phones stay one segment |
| test_per_scores_and_writes_alignment | PER on a known pair, plus its alignment file |
| test_per_is_zero_for_identical_transcripts | Identical input scores 0 |
| test_per_ignores_tokenization_differences | Tokenization does not matter |
| test_pfer_is_below_per_for_near_miss | Feature distance gives partial credit |
| test_pfer_is_zero_for_identical_transcripts | Identical input scores 0 |
| test_pfer_handles_unscorable_reference | Empty reference yields 0 |
| test_metrics_reject_misaligned_scp | iter_inputs enforces matching utterance ids |
| test_metrics_require_panphon | A clear error when the optional dependency is absent |
"""

from pathlib import Path

import pytest

import espnet3.systems.pr.metrics.scoring_utils as scoring_utils
from espnet3.systems.pr.metrics.per import PER
from espnet3.systems.pr.metrics.pfer import PFER
from espnet3.systems.pr.metrics.scoring_utils import clean_ipa_text, segment_ipa

panphon = pytest.importorskip("panphon", reason="panphon is an optional dependency")


def _build_inputs(tmp_path: Path, ref_lines, hyp_lines) -> dict:
    ref_path = tmp_path / "ref.scp"
    hyp_path = tmp_path / "hyp.scp"
    ref_path.write_text("\n".join(ref_lines), encoding="utf-8")
    hyp_path.write_text("\n".join(hyp_lines), encoding="utf-8")
    return {"ref": ref_path, "hyp": hyp_path}


def test_clean_ipa_text_normalizes():
    # Spaces and the slash separators a model may emit are both removed, and
    # Latin g is rewritten as the IPA glyph so both sides compare equal.
    assert clean_ipa_text("ð ɪ s") == "ðɪs"
    assert clean_ipa_text("ð/ɪ/s") == "ðɪs"
    assert clean_ipa_text("go") == "ɡo"
    assert clean_ipa_text("") == ""


def test_clean_ipa_text_strips_angle_brackets():
    # Angle brackets are ASCII punctuation, so an unremoved special token would
    # survive as bare letters and be scored as phones.
    assert clean_ipa_text("<unk>") == "unk"


def test_segment_ipa_keeps_diacritics_together():
    assert segment_ipa("t͡ʃaː") == ["t͡ʃ", "aː"]
    assert segment_ipa("ðɪs") == ["ð", "ɪ", "s"]
    assert segment_ipa("") == []


def test_per_scores_and_writes_alignment(tmp_path: Path):
    # One substitution against a three-phone reference.
    data = _build_inputs(tmp_path, ["utt1 ðɪs"], ["utt1 ðɪz"])

    result = PER()(data, "test", tmp_path)

    assert result == {"PER": pytest.approx(33.3)}
    alignment_path = tmp_path / "test" / "per_alignment"
    assert alignment_path.exists()
    assert alignment_path.read_text().strip() != ""


def test_per_is_zero_for_identical_transcripts(tmp_path: Path):
    data = _build_inputs(tmp_path, ["utt1 ðɪs"], ["utt1 ðɪs"])
    assert PER()(data, "test", tmp_path) == {"PER": 0.0}


def test_per_ignores_tokenization_differences(tmp_path: Path):
    # The reference is space-separated and the hypothesis slash-separated; both
    # are re-segmented, so they must score as a perfect match.
    data = _build_inputs(tmp_path, ["utt1 ð ɪ s"], ["utt1 ð/ɪ/s"])
    assert PER()(data, "test", tmp_path) == {"PER": 0.0}


def test_pfer_is_below_per_for_near_miss(tmp_path: Path):
    # s and z differ only in voicing, so the feature distance is a fraction of
    # the flat cost PER charges.
    data = _build_inputs(tmp_path, ["utt1 ðɪs"], ["utt1 ðɪz"])

    pfer = PFER()(data, "test", tmp_path)["PFER"]
    per = PER()(data, "test", tmp_path)["PER"]

    assert 0.0 < pfer < per


def test_pfer_is_zero_for_identical_transcripts(tmp_path: Path):
    data = _build_inputs(tmp_path, ["utt1 ðɪs"], ["utt1 ðɪs"])
    assert PFER()(data, "test", tmp_path) == {"PFER": 0.0}


def test_pfer_handles_unscorable_reference(tmp_path: Path):
    # "..." normalizes to the empty string, so there is no phone to divide by.
    data = _build_inputs(tmp_path, ["utt1 ..."], ["utt1 ..."])
    assert PFER()(data, "test", tmp_path) == {"PFER": 0.0}


@pytest.mark.parametrize("metric_cls", [PER, PFER])
def test_metrics_reject_misaligned_scp(tmp_path: Path, metric_cls):
    data = _build_inputs(tmp_path, ["utt1 ðɪs"], ["utt2 ðɪs"])
    with pytest.raises(AssertionError):
        metric_cls()(data, "test", tmp_path)


def test_metrics_require_panphon(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(scoring_utils, "panphon", None)
    monkeypatch.setattr(scoring_utils, "_DISTANCE", None)
    data = _build_inputs(tmp_path, ["utt1 ðɪs"], ["utt1 ðɪz"])

    with pytest.raises(RuntimeError, match=r"espnet\[pr\]"):
        PFER()(data, "test", tmp_path)

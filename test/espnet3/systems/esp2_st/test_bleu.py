"""Tests for the ST BLEU metric.

sacrebleu is an extra rather than a base dependency, so each test that needs a
score is guarded the way test_metrics.py guards jiwer: when the package is
absent the metric must raise a named RuntimeError instead of failing
obscurely.
"""

from pathlib import Path

import pytest

from espnet3.systems.esp2_st.metrics.bleu import BLEU

try:
    import sacrebleu  # noqa: F401

    HAS_SACREBLEU = True
except ImportError:
    HAS_SACREBLEU = False

needs_sacrebleu = pytest.mark.skipif(
    not HAS_SACREBLEU, reason="sacrebleu is required for this test"
)


def _build_inputs(tmp_path: Path, ref_lines, hyp_lines) -> dict[str, Path]:
    ref_path = tmp_path / "ref.scp"
    hyp_path = tmp_path / "hyp.scp"
    ref_path.write_text("\n".join(ref_lines), encoding="utf-8")
    hyp_path.write_text("\n".join(hyp_lines), encoding="utf-8")
    return {"ref": ref_path, "hyp": hyp_path}


def test_bleu_requires_sacrebleu(tmp_path: Path, monkeypatch):
    import espnet3.systems.esp2_st.metrics.bleu as bleu_module

    monkeypatch.setattr(bleu_module, "sacrebleu", None)
    data = _build_inputs(tmp_path, ["utt1 hallo welt"], ["utt1 hallo welt"])

    with pytest.raises(RuntimeError, match="sacrebleu is required"):
        BLEU()(data, "test", tmp_path)


def test_bleu_rejects_sacrebleu_1x_api(tmp_path: Path, monkeypatch):
    # In sacrebleu 1.x the name "BLEU" is a namedtuple, not a scorer class.
    import collections

    import espnet3.systems.esp2_st.metrics.bleu as bleu_module

    class Fake:
        BLEU = collections.namedtuple("BLEU", "score")(1.0)
        __version__ = "1.5.1"

    monkeypatch.setattr(bleu_module, "sacrebleu", Fake)
    data = _build_inputs(tmp_path, ["utt1 hallo welt"], ["utt1 hallo welt"])

    with pytest.raises(RuntimeError, match="sacrebleu >= 2.0.0"):
        BLEU()(data, "test", tmp_path)


@needs_sacrebleu
def test_bleu_perfect_match_scores_100(tmp_path: Path):
    lines = ["utt1 das ist ein test", "utt2 hallo welt"]
    data = _build_inputs(tmp_path, lines, lines)

    result = BLEU(chrf=False, ter=False)(data, "test", tmp_path)

    assert result["BLEU"] == 100.0
    assert result["BLEU_1gram_prec"] == 100.0
    assert result["BLEU_brevity_penalty"] == 1.0


@needs_sacrebleu
def test_bleu_reports_chrf_and_ter_like_st_sh(tmp_path: Path):
    # st.sh scores `sacrebleu -m bleu chrf ter`, so all three must be present
    # for both passes.
    data = _build_inputs(
        tmp_path, ["utt1 das ist ein kurzer test"], ["utt1 das ist ein test"]
    )

    result = BLEU()(data, "test", tmp_path)

    for key in ("BLEU", "chrF2", "TER", "BLEU_lc", "chrF2_lc", "TER_lc"):
        assert key in result, key
    # TER is an edit RATE, so a hypothesis missing a word costs > 0.
    assert result["TER"] > 0


@needs_sacrebleu
def test_bleu_can_report_bleu_only(tmp_path: Path):
    data = _build_inputs(tmp_path, ["utt1 hallo welt"], ["utt1 hallo welt"])

    result = BLEU(chrf=False, ter=False)(data, "test", tmp_path)

    assert "chrF2" not in result
    assert "TER" not in result


@needs_sacrebleu
def test_bleu_lowercase_pass_strips_punctuation(tmp_path: Path):
    """st.sh runs remove_punctuation.pl before `sacrebleu -lc` (st.sh:1609)."""
    # Hypothesis differs from the reference ONLY in case and punctuation, so
    # the case-sensitive pass must be below 100 and the _lc pass exactly 100.
    data = _build_inputs(
        tmp_path, ["utt1 Das ist ein Test."], ["utt1 das ist ein test"]
    )

    result = BLEU(chrf=False, ter=False)(data, "test", tmp_path)

    assert result["BLEU"] < 100.0
    assert result["BLEU_lc"] == 100.0


@needs_sacrebleu
def test_bleu_lowercase_pass_can_keep_punctuation(tmp_path: Path):
    # With the st.sh punctuation step disabled, the trailing period survives
    # and the lowercase pass no longer matches exactly.
    data = _build_inputs(
        tmp_path, ["utt1 Das ist ein Test."], ["utt1 das ist ein test"]
    )

    result = BLEU(chrf=False, ter=False, lc_remove_punctuation=False)(
        data, "test", tmp_path
    )

    assert result["BLEU_lc"] < 100.0


@needs_sacrebleu
def test_bleu_can_skip_the_lowercase_pass(tmp_path: Path):
    data = _build_inputs(tmp_path, ["utt1 hallo welt"], ["utt1 hallo welt"])

    result = BLEU(also_lowercase=False, chrf=False, ter=False)(data, "test", tmp_path)

    assert "BLEU" in result
    assert not any(key.endswith("_lc") for key in result)


@needs_sacrebleu
def test_bleu_keeps_empty_hypotheses_empty(tmp_path: Path):
    """An undecodable utterance must not be handed a free n-gram match."""
    # "utt2" has no text, as src/inference.py writes for a TooShortUttError.
    data = _build_inputs(
        tmp_path, ["utt1 hallo welt", "utt2 guten tag"], ["utt1 hallo welt", "utt2 "]
    )

    result = BLEU(chrf=False, ter=False)(data, "test", tmp_path)

    assert result["BLEU"] < 100.0


@needs_sacrebleu
def test_bleu_writes_detail_with_signature(tmp_path: Path):
    data = _build_inputs(tmp_path, ["utt1 hallo welt"], ["utt1 hallo welt"])

    BLEU()(data, "tst-COMMON", tmp_path)

    detail = (tmp_path / "tst-COMMON" / "bleu_detail").read_text(encoding="utf-8")
    # The signature is what makes a BLEU number reproducible across papers.
    assert "nrefs:1" in detail
    assert "sentences: 1" in detail


@needs_sacrebleu
def test_bleu_respects_custom_keys(tmp_path: Path):
    ref_path = tmp_path / "gold.scp"
    hyp_path = tmp_path / "pred.scp"
    # Four tokens minimum: corpus BLEU needs a 4-gram to exist, or the
    # geometric mean over the precisions is 0 regardless of the match.
    ref_path.write_text("utt1 das ist ein test", encoding="utf-8")
    hyp_path.write_text("utt1 das ist ein test", encoding="utf-8")

    result = BLEU(ref_key="gold", hyp_key="pred", chrf=False, ter=False)(
        {"gold": ref_path, "pred": hyp_path}, "test", tmp_path
    )

    assert result["BLEU"] == 100.0

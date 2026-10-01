"""OWSM metrics, which must split one test set across two tasks."""

from __future__ import annotations

from pathlib import Path

import pytest

from espnet3.systems.owsm.metrics import CER, TER, WER
from espnet3.systems.owsm.metrics.normalization import strip_markup

ASR = "<eng><asr><0.00> the quick brown fox<2.50>"
DE = "<eng><st_deu><0.00> der schnelle braune Fuchs<2.50>"


def _scp(tmp_path: Path, name: str, rows: list[tuple[str, str]]) -> Path:
    path = tmp_path / f"{name}.scp"
    path.write_text(
        "".join(f"{utt} {text}\n" for utt, text in rows), encoding="utf-8"
    )
    return path


def _data(tmp_path, refs, hyps):
    utts = [f"utt{i}" for i in range(len(refs))]
    return {
        "ref": _scp(tmp_path, "ref", list(zip(utts, refs))),
        "hyp": _scp(tmp_path, "hyp", list(zip(utts, hyps))),
    }


def test_markup_is_stripped_before_scoring():
    assert strip_markup(ASR) == "the quick brown fox"
    assert strip_markup("<eng><asr><notimestamps> no times") == "no times"


def test_tags_are_removed_so_a_perfect_transcription_scores_zero(tmp_path):
    """The hypothesis never carries tags, so a tagged reference cannot match."""
    pytest.importorskip("jiwer")
    data = _data(tmp_path, [ASR], ["the quick brown fox"])

    assert WER()(data, "dev", tmp_path) == {"WER": 0.0}


def test_remove_tags_false_reproduces_s2t_sh(tmp_path):
    """s2t.sh asks for tag removal and cannot achieve it on OWSM text.

    Kept only so the two numbers can be compared; a perfect transcription
    scores a large WER this way.
    """
    pytest.importorskip("jiwer")
    data = _data(tmp_path, [ASR], ["the quick brown fox"])

    assert WER(remove_tags=False)(data, "dev", tmp_path)["WER"] > 0


def test_every_row_is_scored_including_translations(tmp_path):
    """s2t.sh does not split by task; a translation is scored as a transcript."""
    pytest.importorskip("jiwer")
    asr_dir = tmp_path / "asr_only"
    asr_dir.mkdir()
    # The ASR row is perfect; the translation row is entirely wrong.
    both = _data(tmp_path, [ASR, DE], ["the quick brown fox", "voellig falsch hier"])
    asr_only = _data(asr_dir, [ASR], ["the quick brown fox"])

    metric = WER()
    assert metric(asr_only, "dev", tmp_path) == {"WER": 0.0}
    # Skipping the ST row would keep this at 0.
    assert metric(both, "dev", tmp_path)["WER"] > 0


def test_cer_counts_characters_not_words(tmp_path):
    pytest.importorskip("jiwer")
    # One wrong letter in a four-word line: small CER, large WER.
    data = _data(tmp_path, [ASR], ["the quick brown fix"])

    cer = CER()(data, "dev", tmp_path)["CER"]
    wer = WER()(data, "dev", tmp_path)["WER"]
    assert 0 < cer < wer


def test_ter_counts_bpe_pieces(tmp_path):
    """s2t.sh's ter is a token error rate over subwords, not sacreBLEU's TER."""
    pytest.importorskip("sentencepiece")
    pytest.importorskip("jiwer")
    model = Path("egs3/owsm_v4/owsm/data/bpe_dev_smoke/bpe.model")
    if not model.is_file():
        pytest.skip(f"no tokenizer at {model}; run train_tokenizer first")
    data = _data(tmp_path, [ASR], ["the quick brown fix"])

    ter = TER(bpemodel=model)(data, "dev", tmp_path)["TER"]
    assert ter > 0
    assert (tmp_path / "dev" / "ter_alignment").is_file()


def test_cleaner_is_separate_for_reference_and_hypothesis(tmp_path):
    """s2t.sh keeps --cleaner and --hyp_cleaner apart, so this does too."""
    pytest.importorskip("jiwer")
    metric = WER(cleaner=["whisper_basic"], hyp_cleaner=None)

    assert metric.ref_cleaner is not metric.hyp_cleaner

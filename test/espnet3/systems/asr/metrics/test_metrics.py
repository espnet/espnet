from pathlib import Path

import pytest
import sentencepiece as spm

import espnet3.systems.asr.metrics.cer as cer_module
import espnet3.systems.asr.metrics.wer as wer_module
from espnet3.systems.asr.metrics.cer import CER
from espnet3.systems.asr.metrics.ter import TER
from espnet3.systems.asr.metrics.wer import WER


def _build_inputs(tmp_path: Path, ref_lines, hyp_lines) -> dict[str, Path]:
    ref_path = tmp_path / "ref.scp"
    hyp_path = tmp_path / "hyp.scp"
    ref_path.write_text("\n".join(ref_lines), encoding="utf-8")
    hyp_path.write_text("\n".join(hyp_lines), encoding="utf-8")
    return {"ref": ref_path, "hyp": hyp_path}


def test_wer_writes_alignment_and_score(tmp_path: Path):
    metric = WER()
    data = _build_inputs(tmp_path, ["utt1 hello world"], ["utt1 hello word"])

    try:
        import jiwer  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match=r"espnet\[asr\]"):
            metric(data, "test-clean", tmp_path)
    else:
        result = metric(data, "test-clean", tmp_path)

        assert result == {"WER": 50.0}
        alignment_path = tmp_path / "test-clean" / "wer_alignment"
        assert alignment_path.exists()
        assert alignment_path.read_text().strip() != ""


def test_cer_writes_alignment_and_score(tmp_path: Path):
    metric = CER()
    data = _build_inputs(tmp_path, ["utt1 abc"], ["utt1 axc"])

    try:
        import jiwer  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match=r"espnet\[asr\]"):
            metric(data, "test-other", tmp_path)
    else:
        result = metric(data, "test-other", tmp_path)

        assert result == {"CER": 33.33}
        alignment_path = tmp_path / "test-other" / "cer_alignment"
        assert alignment_path.exists()
        assert alignment_path.read_text().strip() != ""


def test_wer_rejects_unaligned_utt_ids(tmp_path: Path):
    metric = WER()
    data = _build_inputs(tmp_path, ["utt1 hello"], ["utt2 hello"])

    try:
        import jiwer  # noqa: F401
    except ImportError:
        pytest.skip("jiwer is required for this test")
    else:
        with pytest.raises(AssertionError, match="UID mismatch"):
            metric(data, "test-clean", tmp_path)


def test_wer_clean_uses_placeholder_for_empty_text():
    assert WER()._clean("   ") == "."


def test_cer_clean_uses_placeholder_for_empty_text():
    assert CER()._clean("   ") == "."


def test_wer_requires_jiwer(monkeypatch):
    monkeypatch.setattr(wer_module, "jiwer", None)

    with pytest.raises(RuntimeError, match="jiwer is required to compute WER"):
        WER()._ensure_jiwer()


def test_cer_requires_jiwer(monkeypatch):
    monkeypatch.setattr(cer_module, "jiwer", None)

    with pytest.raises(RuntimeError, match="jiwer is required to compute CER"):
        CER()._ensure_jiwer()


@pytest.fixture
def tiny_bpemodel(tmp_path: Path) -> str:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "\n".join(["hello world", "the cat sat", "a dog ran", "hello there"] * 8),
        encoding="utf-8",
    )
    prefix = tmp_path / "bpe"
    spm.SentencePieceTrainer.Train(
        input=str(corpus),
        model_prefix=str(prefix),
        vocab_size=40,
        model_type="bpe",
        character_coverage=1.0,
    )
    return f"{prefix}.model"


@pytest.mark.execution_timeout(30)
def test_ter_zero_when_identical(tmp_path: Path, tiny_bpemodel: str):
    metric = TER(bpemodel=tiny_bpemodel)
    data = _build_inputs(tmp_path, ["utt1 hello world"], ["utt1 hello world"])
    try:
        import jiwer  # noqa: F401
    except ImportError:
        pytest.skip("jiwer not installed")
    result = metric(data, "test-clean", tmp_path)
    assert result == {"TER": 0.0}
    assert (tmp_path / "test-clean" / "ter_alignment").exists()


@pytest.mark.execution_timeout(30)
def test_ter_positive_when_different(tmp_path: Path, tiny_bpemodel: str):
    metric = TER(bpemodel=tiny_bpemodel)
    data = _build_inputs(tmp_path, ["utt1 hello world"], ["utt1 the cat sat"])
    try:
        import jiwer  # noqa: F401
    except ImportError:
        pytest.skip("jiwer not installed")
    result = metric(data, "test-clean", tmp_path)
    assert result["TER"] > 0.0


@pytest.mark.execution_timeout(30)
def test_ter_requires_jiwer(tmp_path: Path, tiny_bpemodel: str, monkeypatch):
    import espnet3.systems.asr.metrics.ter as ter_module

    monkeypatch.setattr(ter_module, "jiwer", None)
    metric = TER(bpemodel=tiny_bpemodel)
    data = _build_inputs(tmp_path, ["utt1 hello"], ["utt1 hello"])
    with pytest.raises(RuntimeError, match=r"espnet\[asr\]"):
        metric(data, "test-clean", tmp_path)

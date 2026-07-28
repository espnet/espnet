from pathlib import Path

import pytest

import espnet3.systems.asr.metrics.cer as cer_module
import espnet3.systems.asr.metrics.wer as wer_module
from espnet3.systems.asr.metrics.cer import CER
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

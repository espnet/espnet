from pathlib import Path

from espnet3.systems.asr.metrics.cer import CER
from espnet3.systems.asr.metrics.wer import WER


def test_wer_writes_alignment_and_score(tmp_path: Path):
    metric = WER()
    data = {"ref": ["hello world"], "hyp": ["hello word"]}

    result = metric(data, "test-clean", tmp_path)

    assert result == {"WER": 50.0}
    alignment_path = tmp_path / "test-clean" / "wer_alignment"
    assert alignment_path.exists()
    assert alignment_path.read_text().strip() != ""


def test_cer_writes_alignment_and_score(tmp_path: Path):
    metric = CER()
    data = {"ref": ["abc"], "hyp": ["axc"]}

    result = metric(data, "test-other", tmp_path)

    assert result == {"CER": 33.33}
    alignment_path = tmp_path / "test-other" / "cer_alignment"
    assert alignment_path.exists()
    assert alignment_path.read_text().strip() != ""

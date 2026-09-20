"""Check source scoring conventions against the real optional SCTK binary."""

import json
import shutil

import pytest

from espnet3.systems.asr.metrics.sclite import ScliteErrorRate


@pytest.mark.skipif(shutil.which("sclite") is None, reason="SCTK is not installed")
@pytest.mark.parametrize("token_type,expected", [("word", 50.0), ("char", 44.44)])
def test_case_empty_and_whitespace(tmp_path, token_type, expected):
    """Preserve native case folding, empty hypotheses and field normalization."""
    reference, hypothesis = tmp_path / "ref.scp", tmp_path / "hyp.scp"
    reference.write_text("u1 A  B\nu2 A B\nu3 A B\n")
    hypothesis.write_text("u1 a b\nu2 A C\nu3\n")
    metric = ScliteErrorRate(token_type)
    assert metric({"ref": reference, "hyp": hypothesis}, "test", tmp_path) == {
        metric.name: expected
    }
    output = tmp_path / "test" / f"score_{metric.name.lower()}"
    counts = json.loads((output / "counts.json").read_text())
    assert counts["substitutions"] == 1
    assert counts["insertions"] == 0
    assert counts["deletions"] == (2 if token_type == "word" else 3)
    assert (output / "hyp.trn").read_text().splitlines()[-1] == " (spk-2)"


def test_missing_sctk(tmp_path):
    """Fail with installation guidance instead of changing scoring algorithms."""
    with pytest.raises(RuntimeError, match="SCTK sclite"):
        ScliteErrorRate(sclite="missing-sclite-executable")({}, "test", tmp_path)


@pytest.mark.parametrize("options", [{"token_type": "phone"}, {"token_type": "bpe"}])
def test_invalid_tokenizer(options):
    """Reject unsupported tokenization before reading any inputs."""
    with pytest.raises(ValueError):
        ScliteErrorRate(**options)

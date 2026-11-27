import numpy as np
import pytest
import soundfile as sf

from egs3.wham.snoar.dataset.builder import WHAMBuilder, is_x1_utterance
from espnet3.systems.snoar.src.data.manifest import read_manifest


def _tree(root, n=8):
    for split in ("tr", "cv", "tt"):
        for sub in ("mix_single_anechoic", "s1_anechoic", "noise"):
            (root / split / sub).mkdir(parents=True)
            for i in range(n):
                sf.write(
                    root / split / sub / f"u{split}{i}.wav",
                    np.zeros(1600, np.float32),
                    16000,
                )


def test_build_writes_disjoint_halves(tmp_path, monkeypatch):
    _tree(tmp_path / "wham")
    monkeypatch.setenv("WHAM_ROOT", str(tmp_path / "wham"))
    monkeypatch.setenv("WHAM_OUTPUT", str(tmp_path / "data"))
    builder = WHAMBuilder()
    assert builder.is_source_prepared() and not builder.is_built()
    builder.build()
    assert builder.is_built()
    out = tmp_path / "data" / "manifest"
    x1 = {r["id"] for r in read_manifest(out / "train.jsonl")}
    donors = {r["group"] for r in read_manifest(out / "noise_wham.jsonl")}
    assert x1 and donors and not x1 & donors and len(x1 | donors) == 8
    test = read_manifest(out / "test.jsonl")
    assert len(test) == 8 and test[0]["speech_ref"].endswith("s1_anechoic/utt0.wav")
    assert all("speech_ref" not in r for r in read_manifest(out / "train.jsonl"))


def test_missing_source_explains_how_to_get_it(monkeypatch):
    monkeypatch.delenv("WHAM_ROOT", raising=False)
    builder = WHAMBuilder()
    assert not builder.is_source_prepared()
    with pytest.raises(FileNotFoundError, match="WHAM_ROOT"):
        builder.prepare_source()
    assert is_x1_utterance("a", 1.0, "s") and not is_x1_utterance("a", 0.0, "s")

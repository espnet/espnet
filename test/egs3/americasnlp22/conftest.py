"""Shared fixtures for the AmericasNLP 2022 recipe tests (no network access)."""

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

# `test/egs3` is not an __init__.py package, so make the repo root importable
# when pytest does not already put it on sys.path.
_REPO_ROOT = str(Path(__file__).resolve().parents[3])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

SAMPLE_RATE = 16000


def write_wav(path: Path, duration_sec: float) -> None:
    """Write a silent wav of the given duration."""
    samples = np.zeros(int(SAMPLE_RATE * duration_sec), dtype=np.float32)
    sf.write(str(path), samples, SAMPLE_RATE)


@pytest.fixture
def corpus_factory():
    """Return a factory building a minimal ``<language>/{train,dev}`` corpus.

    The layout mirrors the shared-task archives: one directory per language
    holding ``train/`` and ``dev/`` splits, each with a ``meta.tsv``
    (``wav | source_processed | source_raw | target_raw``) and one wav per
    utterance.
    """

    def make_corpus(root: Path, language: str = "Bribri") -> Path:
        train_utts = [f"bribri{i:06d}" for i in range(2)]
        dev_utts = [f"bribri{i:06d}" for i in range(2, 3)]
        for split, utt_ids in (("train", train_utts), ("dev", dev_utts)):
            split_dir = root / language / split
            split_dir.mkdir(parents=True, exist_ok=True)
            lines = ["wav\tsource_processed\tsource_raw\ttarget_raw"]
            for i, utt_id in enumerate(utt_ids):
                write_wav(split_dir / f"{utt_id}.wav", duration_sec=0.5 + i * 0.1)
                lines.append(
                    f"{utt_id}.wav\tprocessed {i}\traw text {i}\ttranslation {i}"
                )
            (split_dir / "meta.tsv").write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )
        return root / language

    return make_corpus

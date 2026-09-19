"""Tests for the AISHELL-1 recipe's raw-directory dataset adapter."""

from pathlib import Path

import numpy as np
import soundfile as sf

from egs3.aishell.asr.dataset.builder import AishellBuilder
from egs3.aishell.asr.dataset.dataset import AishellDataset
from egs3.aishell.asr.src.tokenizer import gather_training_text


def _make_aishell_root(tmp_path: Path) -> Path:
    """Create a minimal AISHELL-1 directory tree for one utterance per split."""
    source_root = tmp_path / "data_aishell"
    transcript_lines = []
    for split in ("train", "dev", "test"):
        utt_id = f"BAC009S0002W{len(transcript_lines):04d}"
        audio_dir = source_root / "wav" / split / "S0002"
        audio_dir.mkdir(parents=True)
        sf.write(audio_dir / f"{utt_id}.wav", np.zeros(160, dtype=np.float32), 16000)
        transcript_lines.append(f"{utt_id} 测试 文本")

    transcript_path = source_root / "transcript" / "aishell_transcript_v0.8.txt"
    transcript_path.parent.mkdir()
    transcript_path.write_text("\n".join(transcript_lines), encoding="utf-8")
    return source_root


def test_aishell_builder_and_dataset(tmp_path: Path) -> None:
    """The recipe validates the raw layout and returns mono ASR samples."""
    source_root = _make_aishell_root(tmp_path)
    builder = AishellBuilder()

    assert builder.is_source_prepared(tmp_path, source_dir=source_root)
    assert builder.is_built(tmp_path, source_dir=source_root)

    dataset = AishellDataset("train", recipe_dir=tmp_path, source_dir=source_root)
    sample = dataset[0]

    assert sample["text"] == "测试文本"
    assert sample["speech"].dtype == np.float32
    assert sample["speech"].ndim == 1
    assert set(sample) == {"speech", "text"}


def test_gather_training_text_uses_only_the_train_split(tmp_path: Path) -> None:
    """Tokenizer text excludes transcripts belonging only to dev and test audio."""
    source_root = _make_aishell_root(tmp_path)

    assert gather_training_text(tmp_path, source_dir=source_root) == ["测试文本"]

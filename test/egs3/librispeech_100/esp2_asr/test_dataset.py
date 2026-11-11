"""Tests for the LibriSpeech Hugging Face dataset cache."""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf

from egs3.librispeech_100.esp2_asr.dataset.builder import LibriSpeech100Builder
from egs3.librispeech_100.esp2_asr.dataset.dataset import LibriSpeech100Dataset


def _create_source_split(source_root, split, *, include_audio=True):
    """Create a minimal raw LibriSpeech split with one transcript row."""
    speaker_dir = source_root / split / "1" / "2"
    speaker_dir.mkdir(parents=True, exist_ok=True)
    (speaker_dir / "1-2.trans.txt").write_text(
        "1-2-0000 HELLO WORLD\n", encoding="utf-8"
    )
    if include_audio:
        stream = io.BytesIO()
        sf.write(
            stream,
            np.array([0.0, 0.25, -0.25], dtype=np.float32),
            16000,
            format="FLAC",
        )
        (speaker_dir / "1-2-0000.flac").write_bytes(stream.getvalue())


def _create_source_tree(source_root):
    """Create all raw corpus splits needed by the recipe builder."""
    for split in (
        "train-clean-100",
        "dev-clean",
        "dev-other",
        "test-clean",
        "test-other",
    ):
        _create_source_split(source_root, split)


def test_builder_builds_path_text_hf_datasets_for_all_splits(tmp_path):
    """Save only audio paths and transcripts in each configured HF split."""
    pytest.importorskip("datasets")
    source_root = tmp_path / "source" / "LibriSpeech"
    _create_source_tree(source_root)
    recipe_root = tmp_path / "recipe"
    recipe_root.mkdir()
    builder = LibriSpeech100Builder()

    assert builder.is_source_prepared(recipe_root, source_root) is True
    assert builder.is_built(recipe_root) is False
    builder.prepare_source(recipe_root, source_root)
    builder.build(recipe_root, source_root)

    assert builder.is_built(recipe_root) is True
    train_dataset = LibriSpeech100Dataset("train-clean-100", recipe_dir=recipe_root)
    assert train_dataset._dataset.column_names == ["audio", "text"]
    assert len(train_dataset) == 1
    sample = train_dataset[0]
    assert sample["speech"].dtype == np.float32
    np.testing.assert_allclose(sample["speech"], [0.0, 0.25, -0.25], atol=1e-4)
    assert sample["text"] == "HELLO WORLD"

    cache_root = recipe_root / "data/huggingface/librispeech_asr"
    for split in (
        "train-clean-100",
        "dev-clean",
        "dev-other",
        "test-clean",
        "test-other",
    ):
        split_dataset = LibriSpeech100Dataset(split, recipe_dir=recipe_root)
        assert len(split_dataset) == 1
        assert (cache_root / split / "state.json").is_file()


def test_builder_detects_missing_raw_splits_without_creating_cache(tmp_path):
    """Report incomplete raw data without downloading or writing cache files."""
    source_root = tmp_path / "source" / "LibriSpeech"
    _create_source_split(source_root, "train-clean-100")
    recipe_root = tmp_path / "recipe"
    recipe_root.mkdir()
    builder = LibriSpeech100Builder()

    assert builder.is_source_prepared(recipe_root, source_root) is False
    with pytest.raises(FileNotFoundError, match="missing"):
        builder.prepare_source(recipe_root, source_root)
    assert not (recipe_root / "data").exists()


def test_dataset_reports_missing_waveform_referenced_by_transcript(tmp_path):
    """Keep index creation cheap and report a missing waveform when read."""
    pytest.importorskip("datasets")
    source_root = tmp_path / "source" / "LibriSpeech"
    for split in (
        "train-clean-100",
        "dev-clean",
        "dev-other",
        "test-clean",
        "test-other",
    ):
        _create_source_split(
            source_root,
            split,
            include_audio=split != "train-clean-100",
        )
    recipe_root = tmp_path / "recipe"
    recipe_root.mkdir()

    builder = LibriSpeech100Builder()
    builder.build(recipe_root, source_root)
    dataset = LibriSpeech100Dataset("train-clean-100", recipe_dir=recipe_root)
    with pytest.raises(FileNotFoundError, match="audio file is missing"):
        dataset[0]


def test_dataset_rejects_unsupported_split_before_loading_cache():
    """Reject invalid split names before looking for saved datasets."""
    with pytest.raises(ValueError, match="Unknown split"):
        LibriSpeech100Dataset("train-other-500")

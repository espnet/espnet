"""Tests for the LibriTTS dataset package of the LACE recipe."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from egs3.libritts.lace.dataset import Dataset, DatasetBuilder
from egs3.libritts.lace.dataset.builder import LibriTTSBuilder
from egs3.libritts.lace.dataset.dataset import LibriTTSCodecDataset
from espnet3.components.data.base_dataset import BaseDataset
from espnet3.components.data.dataset_module import instantiate_dataset_reference

SUBSETS = [
    "train-clean-100",
    "train-clean-360",
    "train-other-500",
    "dev-clean",
    "test-clean",
]


def _write_corpus(recipe_dir: Path, utterances_per_subset: int = 2) -> None:
    """Write a tiny LibriTTS tree under ``<recipe_dir>/downloads`` with markers."""
    root = recipe_dir / "downloads" / "LibriTTS"
    for subset_index, subset in enumerate(SUBSETS):
        speaker, chapter = f"{100 + subset_index}", "1"
        chapter_dir = root / subset / speaker / chapter
        chapter_dir.mkdir(parents=True)
        for utt in range(utterances_per_subset):
            stem = f"{speaker}_{chapter}_{utt:06d}_000000"
            sf.write(chapter_dir / f"{stem}.wav", np.zeros(2400, np.float32), 24000)
            (chapter_dir / f"{stem}.normalized.txt").write_text(f"text {utt}\n")
        (root / subset / ".complete").touch()


def test_package_exports_the_builder_and_dataset():
    assert DatasetBuilder is LibriTTSBuilder
    assert Dataset is LibriTTSCodecDataset
    # The dataset contract: a BaseDataset with declared item fields.
    assert issubclass(LibriTTSCodecDataset, BaseDataset)
    assert [(f.name, f.kind) for f in LibriTTSCodecDataset.fields] == [
        ("audio", "audio")
    ]


def test_builder_writes_manifests_and_dataset_reads_them(tmp_path):
    _write_corpus(tmp_path)
    builder = LibriTTSBuilder()
    assert builder.is_source_prepared(recipe_dir=tmp_path)
    assert not builder.is_built(recipe_dir=tmp_path)

    builder.build(recipe_dir=tmp_path)

    assert builder.is_built(recipe_dir=tmp_path)
    manifest = tmp_path / "data" / "manifest"
    assert len((manifest / "train.tsv").read_text().splitlines()) == 6  # 3 subsets
    assert len((manifest / "valid.tsv").read_text().splitlines()) == 2
    assert len((manifest / "test.tsv").read_text().splitlines()) == 2

    train = LibriTTSCodecDataset(split="train", recipe_dir=tmp_path)
    assert len(train) == 6
    assert set(train[0]) == {"audio"}
    assert train[0]["audio"].dtype == np.float32

    test = LibriTTSCodecDataset(split="test", recipe_dir=tmp_path, inference=True)
    assert [f.name for f in test.fields] == ["audio", "utt_id", "wav_path"]
    sample = test[0]
    assert set(sample) == {"audio", "utt_id", "wav_path"}
    assert sample["utt_id"] == "104-1-000000-000000"
    assert Path(sample["wav_path"]).is_file()
    # The recipe's dataset reference (as the stages build it) is accepted.
    built = instantiate_dataset_reference(
        {
            "data_src": "egs3.libritts.lace.dataset",
            "data_src_args": {
                "split": "test",
                "inference": True,
                "recipe_dir": tmp_path,
            },
        }
    )
    assert isinstance(built, LibriTTSCodecDataset) and len(built) == 2


def test_dataset_requires_built_manifests(tmp_path):
    with pytest.raises(RuntimeError, match="not built yet"):
        LibriTTSCodecDataset(split="train", recipe_dir=tmp_path)


def test_source_is_not_prepared_without_markers(tmp_path):
    _write_corpus(tmp_path)
    (tmp_path / "downloads" / "LibriTTS" / "dev-clean" / ".complete").unlink()
    assert not LibriTTSBuilder().is_source_prepared(recipe_dir=tmp_path)

"""Tiny real-audio TSE fixture preparation and sample contracts."""

import numpy as np
import pytest
import soundfile as sf

from egs3.mini_an4.esp2_tse.dataset import Dataset, DatasetBuilder


def test_builder_and_training_samples(tmp_path):
    source = tmp_path / "downloads/an4/wav"
    source.mkdir(parents=True)
    for idx in range(4):
        sf.write(
            source / f"{idx}.sph",
            np.random.default_rng(idx).normal(0, 0.1, 2048),
            16000,
            format="NIST",
        )
    builder = DatasetBuilder()
    assert builder.is_source_prepared(recipe_dir=tmp_path)
    assert not builder.is_built(recipe_dir=tmp_path)
    builder.build(recipe_dir=tmp_path)
    assert builder.is_built(recipe_dir=tmp_path)
    ds = Dataset(split="train", recipe_dir=tmp_path)
    assert len(ds) == 2
    assert "utt_id" not in ds[0]
    evaluation = Dataset(split="test", recipe_dir=tmp_path, include_ids=True)
    assert evaluation[0]["utt_id"]
    with pytest.raises(ValueError, match="split"):
        Dataset(split="invalid", recipe_dir=tmp_path)


def test_missing_source_fails_without_download(tmp_path):
    with pytest.raises(FileNotFoundError, match="downloads.tar.gz"):
        DatasetBuilder().prepare_source(recipe_dir=tmp_path)

"""Verify VOiCES preparation, split membership and artifact integrity."""

import json
import shutil
import tarfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from egs3.voices.asr.dataset import DatasetBuilder
from egs3.voices.asr.dataset import builder as module


def test_checks_are_read_only(tmp_path):
    """Status probes must not create directories or trigger downloads."""
    builder = DatasetBuilder()
    assert not builder.is_source_prepared(recipe_dir=tmp_path)
    assert not builder.is_built(recipe_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_speaker_split_and_source_grouping(corpus):
    """Keep every version of one source in its original speaker's partition."""
    recipe, source = corpus
    builder = DatasetBuilder()
    builder.build(recipe_dir=recipe)
    assert builder.is_built(recipe_dir=recipe)
    assert not builder.is_built(recipe_dir=recipe, source_dir=source / "elsewhere")
    train, valid, test = [
        module.read_manifest(recipe / f"data/manifest/{split}.tsv")
        for split in module.SPLITS
    ]
    assert [len(rows) for rows in (train, valid, test)] == [6, 6, 3]
    assert {row["speaker"] for row in valid} == {"0001", "0002"}
    groups = [{row["source_id"] for row in rows} for rows in (train, valid, test)]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    assert {row["condition"] for row in test} == {"source", "distant"}
    assert (recipe / "data/manifest/tokenizer_train.txt").read_text().splitlines() == [
        row["text"] for row in train
    ]
    marker = recipe / "data/manifest/build.json"
    first = marker.read_bytes()
    builder.build(recipe_dir=recipe)
    assert marker.read_bytes() == first


def test_duration_filter_does_not_filter_test(corpus):
    """Match strict ESPnet2 train/dev duration bounds; retain all test audio."""
    recipe, source = corpus
    train = sorted((source / "source-16k/train").rglob("*.wav"))
    test = next((source / "source-16k/test").rglob("*.wav"))
    sf.write(train[0], np.zeros(1600), 16000)
    sf.write(train[-1], np.zeros(480000), 16000)
    sf.write(test, np.zeros(800), 16000)
    DatasetBuilder().build(recipe_dir=recipe)
    assert len(module.read_manifest(recipe / "data/manifest/train.tsv")) == 5
    assert len(module.read_manifest(recipe / "data/manifest/valid.tsv")) == 5
    assert len(module.read_manifest(recipe / "data/manifest/test.tsv")) == 3
    assert (
        json.loads((recipe / "data/manifest/build.json").read_text())[
            "filtered_train_valid"
        ]
        == 2
    )


def test_reject_incomplete_audio_and_conflicting_transcripts(corpus):
    """Incomplete archives and ambiguous references must fail before training."""
    recipe, source = corpus
    path = next((source / "distant-16k/speech/test").rglob("*.wav"))
    original = path.read_bytes()
    path.unlink()
    with pytest.raises(ValueError, match="Incomplete devkit test"):
        DatasetBuilder().build(recipe_dir=recipe)
    path.write_bytes(original)
    with (source / "references/filename_transcripts").open("a") as stream:
        stream.write(f"0,{path.name},CONFLICT\n")
    with pytest.raises(ValueError, match="conflicting"):
        DatasetBuilder().build(recipe_dir=recipe)
    assert not DatasetBuilder().is_built(recipe_dir=recipe)


def test_audio_and_source_validation(corpus):
    """Reject unsupported audio and missing sources instead of resampling."""
    recipe, source = corpus
    path = next((source / "source-16k/train").rglob("*.wav"))
    sf.write(path, np.zeros(8000), 8000)
    with pytest.raises(ValueError, match="mono 16 kHz"):
        DatasetBuilder().build(recipe_dir=recipe)
    with pytest.raises(FileNotFoundError, match="Incomplete extracted"):
        DatasetBuilder().prepare_source(recipe_dir=recipe, source_dir=source / "absent")


def test_shared_download_and_extraction(corpus, tmp_path, monkeypatch):
    """Exercise the same shared extractor with a small local devkit archive."""
    recipe, source = corpus
    archive = tmp_path / "fixture.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        stream.add(source, arcname="VOiCES_devkit")
    destination = recipe / "other_recipe"
    called = []

    def download(url, path):
        called.append(url)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(archive, path)

    monkeypatch.setattr(module, "download_url", download)
    builder = DatasetBuilder()
    builder.prepare_source(recipe_dir=destination)
    builder.build(recipe_dir=destination)
    assert called == [module._BUILDER_CFG["url"]]
    assert builder.is_built(recipe_dir=destination)
    builder.prepare_source(recipe_dir=destination)
    assert len(called) == 1


def test_missing_clean_source_is_not_silently_accepted(corpus):
    """A complete distant subset also requires every corresponding clean source."""
    recipe, source = corpus
    next((source / "source-16k/train").rglob("*.wav")).unlink()
    with pytest.raises(ValueError, match="Source/distant utterance sets differ"):
        DatasetBuilder().build(recipe_dir=recipe)
    assert not DatasetBuilder().is_built(recipe_dir=recipe)


def test_build_marker_with_real_builder_configuration(tmp_path, monkeypatch):
    """Persist the production YAML settings without replacing their container type."""
    rows = [
        dict(
            utt_id=f"{speaker:04d}_recording",
            path="/unused.wav",
            text="WORDS",
            speaker=f"{speaker:04d}",
            source_id=f"source_{speaker}",
            condition="distant",
            samples=16000,
        )
        for speaker in range(1, 13)
    ]
    monkeypatch.setattr(module, "_load_transcripts", lambda _: {})
    monkeypatch.setattr(
        module,
        "_scan_recordings",
        lambda _, split, __, corpus: rows[:-1] if split == "train" else rows[-1:],
    )
    builder = DatasetBuilder()
    builder.build(recipe_dir=tmp_path)
    metadata = json.loads((tmp_path / "data/manifest/build.json").read_text())
    assert metadata["builder"]["expected_distant_counts"] == {
        "train": 12800,
        "test": 6400,
    }
    assert metadata["counts"] == {"train": 1, "valid": 10, "test": 1}
    assert builder.is_built(recipe_dir=tmp_path)


def test_full_corpus_mode_with_existing_layout(corpus, monkeypatch):
    """Apply the full-corpus mode to a local layout without a large download."""
    recipe, source = corpus
    monkeypatch.setitem(
        module._BUILDER_CFG, "full_distant_counts", {"train": 8, "test": 2}
    )
    builder = DatasetBuilder()
    builder.prepare_source(recipe_dir=recipe, source_dir=source, corpus="full")
    builder.build(recipe_dir=recipe, source_dir=source, corpus="full")
    assert builder.is_built(recipe_dir=recipe, source_dir=source, corpus="full")
    assert not builder.is_built(recipe_dir=recipe, source_dir=source, corpus="devkit")
    assert len(module.read_manifest(recipe / "data/manifest/train.tsv")) == 6


@pytest.mark.parametrize("split", module.SPLITS)
@pytest.mark.parametrize("damage", ["truncate", "replace"])
def test_rebuild_corrupted_manifest(corpus, split, damage):
    """Detect damaged cached rows before reusing them and rebuild from source."""
    recipe, _ = corpus
    builder = DatasetBuilder()
    builder.build(recipe_dir=recipe)
    path = recipe / f"data/manifest/{split}.tsv"
    original = path.read_bytes()
    corrupted = original[: len(original) // 2] if damage == "truncate" else b"changed\n"
    path.write_bytes(corrupted)
    assert not builder.is_built(recipe_dir=recipe)
    assert path.read_bytes() == corrupted  # The probe itself remains read-only.
    builder.build(recipe_dir=recipe)
    assert path.read_bytes() == original
    assert builder.is_built(recipe_dir=recipe)


@pytest.mark.parametrize("hashes", [None, {}, [], {"train": "stale"}])
def test_rebuild_without_valid_manifest_hashes(corpus, hashes):
    """Do not trust markers with absent, malformed or outdated digests."""
    recipe, _ = corpus
    builder = DatasetBuilder()
    builder.build(recipe_dir=recipe)
    marker = recipe / "data/manifest/build.json"
    metadata = json.loads(marker.read_text())
    if hashes is None:
        metadata.pop("manifest_sha256")
    else:
        metadata["manifest_sha256"] = hashes
    marker.write_text(json.dumps(metadata))
    assert not builder.is_built(recipe_dir=recipe)
    builder.build(recipe_dir=recipe)
    assert builder.is_built(recipe_dir=recipe)


@pytest.mark.parametrize("contents", ["{", "null", "[]"])
def test_invalid_build_marker_is_not_reused(corpus, contents):
    """Treat a truncated or structurally invalid build marker as incomplete."""
    recipe, _ = corpus
    builder = DatasetBuilder()
    builder.build(recipe_dir=recipe)
    (recipe / "data/manifest/build.json").write_text(contents)
    assert not builder.is_built(recipe_dir=recipe)
    builder.build(recipe_dir=recipe)
    assert builder.is_built(recipe_dir=recipe)

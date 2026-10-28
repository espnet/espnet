"""Regression tests for full AN4 preparation and migration settings."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from egs3.an4.esp2_asr.dataset import Dataset, DatasetBuilder
from egs3.an4.esp2_asr.dataset.builder import read_manifest
from egs3.an4.esp2_asr.src.tokenizer import gather_training_text

ROOT = Path(__file__).resolve().parents[5]
RECIPE = ROOT / "egs3/an4/esp2_asr"


def test_checks_are_read_only(tmp_path):
    """Preparation checks must leave a fresh directory untouched."""
    builder = DatasetBuilder()
    assert not builder.is_source_prepared(recipe_dir=tmp_path)
    assert not builder.is_built(recipe_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_split_augmentation_and_pcm(corpus):
    """Sort and split before perturbation, preserving unmodified PCM."""
    builder = DatasetBuilder()
    builder.build(recipe_dir=corpus, dev_size=1)
    assert builder.is_built(recipe_dir=corpus, dev_size=1)
    assert not builder.is_built(recipe_dir=corpus, dev_size=2)
    train = Dataset("train", recipe_dir=corpus)
    valid = Dataset("valid", recipe_dir=corpus)
    test = Dataset("test", recipe_dir=corpus)
    assert (len(train), len(valid), len(test)) == (6, 1, 1)
    assert valid.entries[0][0] == "aa-an1-b"
    assert {entry[2] for entry in train.entries} == {"TEXT BB", "TEXT ZZ"}
    assert {entry[2] for entry in test.entries} == {"TEXT TT"}
    assert [entry[0] for entry in train.entries] == [
        "bb-an1-b",
        "sp0.9-bb-an1-b",
        "sp0.9-zz-an1-b",
        "sp1.1-bb-an1-b",
        "sp1.1-zz-an1-b",
        "zz-an1-b",
    ]
    assert gather_training_text(corpus / "data/manifest/train.tsv") == [
        row[2] for row in train.entries
    ]
    for index, (uid, _, _) in enumerate(train.entries):
        sample = train[index]
        assert set(sample) == {"speech", "text"}
        assert sample["speech"].dtype == np.float32
        speed = float(uid.split("-")[0][2:]) if uid.startswith("sp") else 1.0
        assert abs(len(sample["speech"]) - 16000 / speed) <= 1
    source = corpus / "downloads/an4/wav/an4_clstk/aa/an1-aa-b.sph"
    np.testing.assert_array_equal(
        valid[0]["speech"], sf.read(source, dtype="float32")[0]
    )


@pytest.mark.parametrize("empty_speaker", ["aa", "bb"])
def test_empty_transcripts_follow_espnet2_filtering(corpus, empty_speaker):
    """Filter empty valid text after splitting, but preserve test text."""
    source = corpus / "downloads/an4/etc"
    train_path = source / "an4_train.transcription"
    train_path.write_text(
        train_path.read_text().replace(f"TEXT {empty_speaker.upper()}", "")
    )
    test_path = source / "an4_test.transcription"
    test_path.write_text(test_path.read_text().replace("TEXT TT", ""))
    DatasetBuilder().build(recipe_dir=corpus, dev_size=2)
    valid = Dataset("valid", recipe_dir=corpus)
    train = Dataset("train", recipe_dir=corpus)
    test = Dataset("test", recipe_dir=corpus)
    assert len(valid) == 1
    assert len(train) == 3
    assert all(entry[2].strip() for entry in valid.entries + train.entries)
    assert len(test) == 1 and test.entries[0][2] == ""


def test_empty_training_transcript_is_filtered(corpus):
    """Remove all speed variants of an empty training transcript."""
    path = corpus / "downloads/an4/etc/an4_train.transcription"
    path.write_text(path.read_text().replace("TEXT BB", ""))
    DatasetBuilder().build(recipe_dir=corpus, dev_size=1)
    train = Dataset("train", recipe_dir=corpus)
    assert len(train) == 3
    assert all(entry[2] == "TEXT ZZ" for entry in train.entries)


def test_reject_incomplete_source_and_invalid_split(corpus):
    """Fail before publishing invalid or incomplete data."""
    builder = DatasetBuilder()
    with pytest.raises(ValueError, match="dev_size"):
        builder.build(recipe_dir=corpus, dev_size=3)
    assert not builder.is_built(recipe_dir=corpus)
    with pytest.raises(ValueError, match="Unknown AN4 split"):
        Dataset("unrecognized", recipe_dir=corpus)
    with pytest.raises(FileNotFoundError, match="Incomplete AN4"):
        builder.prepare_source(source_dir=corpus / "missing")


def test_manifest_rejects_duplicate_ids(tmp_path):
    """Reject ambiguous utterance identifiers."""
    path = tmp_path / "duplicate.tsv"
    path.write_text("same\t/a.wav\tA\nsame\t/b.wav\tB\n")
    with pytest.raises(ValueError, match="duplicate"):
        read_manifest(path)


def test_test_split_is_not_duration_filtered(corpus):
    """Match ESPnet2 stage 4, which filters train/valid but preserves test audio."""
    path = corpus / "downloads/an4/wav/an4test_clstk/tt/an1-tt-b.sph"
    sf.write(path, np.zeros(800, dtype=np.int16), 16000, format="NIST")
    DatasetBuilder().build(recipe_dir=corpus, dev_size=1)
    assert len(Dataset("test", recipe_dir=corpus)[0]["speech"]) == 800


def test_download_extraction_and_lm_text(corpus, tmp_path, monkeypatch):
    """Exercise a local archive through the real extraction and preparation path."""
    import shutil
    import tarfile

    from egs3.an4.esp2_asr.dataset import builder as module

    archive = tmp_path / "fixture.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        stream.add(corpus / "downloads/an4", arcname="an4")

    def download(url, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(archive, path)

    monkeypatch.setattr(module, "download_url", download)
    output = tmp_path / "extracted"
    builder = DatasetBuilder()
    builder.prepare_source(recipe_dir=output)
    builder.build(recipe_dir=output, dev_size=1)
    assert builder.is_built(recipe_dir=output, dev_size=1)
    assert len((output / "data/lm/train.txt").read_text().splitlines()) == 6
    assert len((output / "data/lm/valid.txt").read_text().splitlines()) == 1

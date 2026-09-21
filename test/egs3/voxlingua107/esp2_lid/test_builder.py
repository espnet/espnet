"""Tests for VoxLingua source validation and recipe-local manifests."""

import shutil
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from omegaconf import OmegaConf

from egs3.voxlingua107.esp2_lid.dataset import Dataset
from egs3.voxlingua107.esp2_lid.dataset import builder as builder_module
from egs3.voxlingua107.esp2_lid.dataset.builder import VoxLingua107Builder
from espnet3.systems.esp2_lid.collect_stats import collect_speech_shapes
from espnet3.utils.config_utils import load_and_merge_config


def _touch_wav(root, language):
    language_dir = root / language
    language_dir.mkdir(parents=True, exist_ok=True)
    (language_dir / "sample.wav").touch()


def test_source_requires_every_training_language(tmp_path, monkeypatch):
    """Require the full source language inventory before building."""
    monkeypatch.setattr(
        builder_module,
        "_ISO3_CODES",
        {"aa": "aaa", "bb": "bbb"},
    )
    _touch_wav(tmp_path / "dev", "aa")
    _touch_wav(tmp_path, "aa")
    _touch_wav(tmp_path, "unknown")
    builder = VoxLingua107Builder()

    assert not builder.is_source_prepared(source_dir=tmp_path)

    _touch_wav(tmp_path, "bb")

    assert builder.is_source_prepared(source_dir=tmp_path)


def test_built_requires_complete_training_metadata(tmp_path, monkeypatch):
    """Reject metadata missing a training language."""
    monkeypatch.setattr(
        builder_module,
        "_ISO3_CODES",
        {"aa": "aaa", "bb": "bbb"},
    )
    metadata_root = tmp_path / "data" / "voxlingua107"
    required = ("manifest.tsv", "utt2lang", "lang2utt", "category2utt")
    for split in ("train", "dev"):
        split_dir = metadata_root / split
        split_dir.mkdir(parents=True)
        for name in required:
            (split_dir / name).write_text("fixture\n", encoding="utf-8")
    category2utt = metadata_root / "train" / "category2utt"
    category2utt.write_text("aaa 0\n", encoding="utf-8")
    builder = VoxLingua107Builder()

    assert not builder.is_built(recipe_dir=tmp_path)

    category2utt.write_text("aaa 0\nbbb 1\n", encoding="utf-8")

    assert builder.is_built(recipe_dir=tmp_path)


def test_build_keeps_metadata_outside_source(tmp_path, monkeypatch):
    """Follow the ASR manifest layout without copying or changing source audio."""
    monkeypatch.setattr(builder_module, "_ISO3_CODES", {"aa": "aaa", "bb": "bbb"})
    source_dir = tmp_path / "source"
    recipe_dir = tmp_path / "recipe"
    for language in ("aa", "bb"):
        for split, filename in (("", "train.wav"), ("dev", "dev.wav")):
            path = source_dir / split / language / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, np.zeros(160), 16000)
    original_files = {path: path.read_bytes() for path in source_dir.rglob("*.wav")}
    builder = VoxLingua107Builder()
    builder.build(source_dir=source_dir, recipe_dir=recipe_dir)

    metadata_root = recipe_dir / "data" / "voxlingua107"
    assert builder.is_built(recipe_dir=recipe_dir)
    assert not (source_dir / "espnet3").exists()
    assert set(source_dir.rglob("*.wav")) == set(original_files)
    assert all(path.read_bytes() == content for path, content in original_files.items())
    dataset = Dataset("train", source_dir=source_dir, recipe_dir=recipe_dir)
    assert len(dataset) == 2
    assert dataset[0]["speech"].shape == (160,)
    assert dataset[1]["lid_labels"] == "bbb"
    assert (metadata_root / "train" / "category2utt").read_text() == "aaa 0\nbbb 1\n"

    other_data = tmp_path / "other_data"
    builder.build(source_dir=source_dir, data_dir=other_data)
    assert builder.is_built(data_dir=other_data)
    assert len(Dataset("dev", source_dir=source_dir, data_dir=other_data)) == 2

    # Evaluation needs only its split, even after training data is removed.
    for language in ("aa", "bb"):
        shutil.rmtree(source_dir / language)
    shutil.rmtree(other_data / "train")
    for path in (other_data / "dev").iterdir():
        if path.name != "manifest.tsv":
            path.unlink()
    evaluation = Dataset("dev", source_dir=source_dir, data_dir=other_data)
    assert len(evaluation) == 2
    assert evaluation[0]["speech"].shape == (160,)
    assert evaluation[1]["lid_labels"] == "bbb"


def test_dataset_requires_explicit_preparation(tmp_path, monkeypatch):
    """Reading a Dataset must not download the corpus or write manifests."""

    def unexpected_write(*args, **kwargs):
        pytest.fail("Dataset constructor started source or metadata preparation")

    monkeypatch.setattr(VoxLingua107Builder, "prepare_source", unexpected_write)
    monkeypatch.setattr(VoxLingua107Builder, "build", unexpected_write)
    with pytest.raises(FileNotFoundError, match="create_dataset"):
        Dataset("dev", source_dir=tmp_path / "source", data_dir=tmp_path / "metadata")
    assert not list(tmp_path.iterdir())


def test_speed_variants_preserve_labels_and_collect_perturbed_shapes(tmp_path):
    """Expose all training variants with accurate lengths and unchanged dev."""
    config = load_and_merge_config(
        Path("egs3/voxlingua107/esp2_lid/conf/training.yaml"),
        config_name="training.yaml",
        default_package="egs3.TEMPLATE.esp2_lid",
        resolve=False,
    )
    config.recipe_dir = str(tmp_path)
    config.dataset.preprocessor = None
    config.dataset.train[0].data_src = "voxlingua107/esp2_lid"
    config.dataset.valid[0].data_src = "egs3.voxlingua107.esp2_lid.dataset"
    config.dataset.train[0].data_src_args.speed_perturb_factors = [0.9, 1.0, 1.1]
    config.dataloader.train.iter_factory.num_workers = 0
    config.dataloader.valid.iter_factory.num_workers = 0
    OmegaConf.resolve(config)
    metadata = tmp_path / "data/voxlingua107"
    audio = tmp_path / "tone.wav"
    tone = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000).astype(np.float32)
    sf.write(audio, tone, 16000)
    original_audio = audio.read_bytes()
    for split in ("train", "dev"):
        directory = metadata / split
        directory.mkdir(parents=True)
        (directory / "manifest.tsv").write_text(f"tone\t{audio}\teng\n")
    train = Dataset("train", data_dir=metadata, speed_perturb_factors=[0.9, 1, 1.1])
    assert len(train) == 3
    for sample, length, frequency in zip(
        (train[index] for index in range(3)), [17778, 16000, 14546], [396, 440, 484]
    ):
        speech = sample["speech"]
        assert speech.shape == (length,)
        assert sample["lid_labels"] == "eng"
        peak = np.fft.rfftfreq(length, 1 / 16000)[np.abs(np.fft.rfft(speech)).argmax()]
        assert abs(peak - frequency) < 2
    dev = Dataset("dev", data_dir=metadata, speed_perturb_factors=[0.9, 1, 1.1])
    assert len(dev) == 1
    np.testing.assert_array_equal(dev[0]["speech"], train[1]["speech"])

    collect_speech_shapes(config)
    stats = tmp_path / "exp/stats"
    assert (stats / "train/speech_shape").read_text() == "0 17778\n1 16000\n2 14546\n"
    assert (stats / "train/category2utt").read_text() == "eng 0 1 2\n"
    assert (stats / "valid/speech_shape").read_text() == "0 16000\n"
    assert audio.read_bytes() == original_audio


def test_interrupted_manifest_build_is_retried(tmp_path, monkeypatch):
    """Even existing files cannot make an interrupted rebuild appear complete."""
    monkeypatch.setattr(builder_module, "_ISO3_CODES", {"aa": "aaa"})
    source = tmp_path / "source"
    for split, name in (("", "train.wav"), ("dev", "dev.wav")):
        path = source / split / "aa" / name
        path.parent.mkdir(parents=True)
        path.touch()
    metadata = tmp_path / "metadata"
    builder = VoxLingua107Builder()
    builder.build(source_dir=source, data_dir=metadata)
    assert builder.is_built(data_dir=metadata)
    write_split = builder_module._write_split

    def interrupt(source_root, metadata_root, split):
        write_split(source_root, metadata_root, split)
        if split == "dev":
            raise RuntimeError("Interrupted metadata build")

    with monkeypatch.context() as context:
        context.setattr(builder_module, "_write_split", interrupt)
        with pytest.raises(RuntimeError, match="Interrupted"):
            builder.build(source_dir=source, data_dir=metadata)
    assert not builder.is_built(data_dir=metadata)
    with pytest.raises(FileNotFoundError, match="create_dataset"):
        Dataset("dev", source_dir=source, data_dir=metadata)
    builder.build(source_dir=source, data_dir=metadata)
    assert builder.is_built(data_dir=metadata)
    (metadata / "dev/manifest.tsv").write_text("")
    assert not builder.is_built(data_dir=metadata)

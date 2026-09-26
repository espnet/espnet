"""Tests for the LibriSpeech 100h kNN-VC recipe dataset and builder."""

import numpy as np
import pytest
import soundfile as sf

from egs3.librispeech_100.knnvc.dataset import Dataset, DatasetBuilder
from egs3.librispeech_100.knnvc.dataset.dataset import (
    FEATURE_SUFFIX,
    LibriSpeech100Dataset,
    _is_valid_utterance,
)

SR = 16000
HOP = 320
# split -> speaker -> chapter -> [(utt index, seconds)]
LAYOUT = {
    "train-clean-100": {
        "11": {"100": [(0, 0.5), (1, 0.7)], "101": [(0, 0.4)]},
        "22": {"200": [(0, 0.6), (1, 0.5), (2, 0.45)]},
    },
    "dev-clean": {
        "33": {"300": [(0, 0.5), (1, 0.6)]},
        "44": {"400": [(0, 0.55)]},
        "55": {"500": [(0, 0.5), (1, 0.5)]},
    },
}


@pytest.fixture
def librispeech(tmp_path):
    """Create a tiny LibriSpeech-shaped tree with flac files and transcripts."""
    root = tmp_path / "corpus" / "LibriSpeech"
    rng = np.random.RandomState(0)
    for split, speakers in LAYOUT.items():
        for speaker, chapters in speakers.items():
            for chapter, utts in chapters.items():
                chapter_dir = root / split / speaker / chapter
                chapter_dir.mkdir(parents=True)
                lines = []
                for index, seconds in utts:
                    utt_id = f"{speaker}-{chapter}-{index:04d}"
                    wav = (rng.randn(int(seconds * SR)) * 0.1).astype(np.float32)
                    sf.write(chapter_dir / f"{utt_id}.flac", wav, SR)
                    lines.append(f"{utt_id} HELLO WORLD {index}\n")
                (chapter_dir / f"{speaker}-{chapter}.trans.txt").write_text(
                    "".join(lines), encoding="utf-8"
                )
    return root


def _count(split):
    return sum(
        len(utts) for chapters in LAYOUT[split].values() for utts in chapters.values()
    )


def test_module_exports_recipe_classes():
    assert Dataset is LibriSpeech100Dataset
    assert DatasetBuilder.__name__ == "LibriSpeech100Builder"


def test_builder_checks_required_splits(tmp_path, librispeech):
    builder = DatasetBuilder()
    assert builder.is_source_prepared(recipe_dir=tmp_path, source_dir=librispeech)
    assert builder.is_built(recipe_dir=tmp_path, source_dir=librispeech.parent)
    builder.build(recipe_dir=tmp_path, source_dir=librispeech)

    assert not builder.is_source_prepared(
        recipe_dir=tmp_path, source_dir=tmp_path / "x"
    )
    with pytest.raises(FileNotFoundError, match="LibriSpeech source not found"):
        builder.prepare_source(recipe_dir=tmp_path, source_dir=tmp_path / "x")

    incomplete = tmp_path / "incomplete" / "LibriSpeech" / "dev-clean"
    incomplete.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="train-clean-100"):
        builder.prepare_source(recipe_dir=tmp_path, source_dir=incomplete.parent)


def test_dataset_validates_arguments(tmp_path, librispeech):
    with pytest.raises(ValueError, match="Unknown split"):
        LibriSpeech100Dataset("nope", recipe_dir=tmp_path, source_dir=librispeech)
    with pytest.raises(ValueError, match="Unknown kind"):
        LibriSpeech100Dataset(
            "dev-clean", kind="mel", recipe_dir=tmp_path, source_dir=librispeech
        )
    with pytest.raises(ValueError, match="features_dir"):
        LibriSpeech100Dataset(
            "dev-clean", kind="vocoder", recipe_dir=tmp_path, source_dir=librispeech
        )


def test_audio_kind_items_and_stage_contract(tmp_path, librispeech):
    dataset = LibriSpeech100Dataset(
        "dev-clean", kind="audio", recipe_dir=tmp_path, source_dir=librispeech
    )
    assert len(dataset) == _count("dev-clean")
    assert dataset.get_pool_key(0) == "33/300"  # official: per chapter directory
    assert dataset.get_feature_name(0) == "dev-clean/33/300/33-300-0000"
    by_speaker = LibriSpeech100Dataset(
        "train-clean-100",
        kind="audio",
        prematch_pool="speaker",
        recipe_dir=tmp_path,
        source_dir=librispeech,
    )
    assert {by_speaker.get_pool_key(i) for i in range(len(by_speaker))} == {"11", "22"}
    with pytest.raises(ValueError, match="prematch_pool"):
        LibriSpeech100Dataset(
            "dev-clean",
            prematch_pool="book",
            recipe_dir=tmp_path,
            source_dir=librispeech,
        )
    item = dataset[0]
    assert set(item) == {"speech"}
    assert item["speech"].dtype == np.float32
    assert item["speech"].shape == (int(0.5 * SR),)


def test_vocoder_kind_reads_aligned_segments(tmp_path, librispeech):
    features_dir = tmp_path / "features"
    audio = LibriSpeech100Dataset(
        "train-clean-100", kind="audio", recipe_dir=tmp_path, source_dir=librispeech
    )
    for idx in range(len(audio)):
        n_frames = len(audio[idx]["speech"]) // HOP
        path = features_dir / (audio.get_feature_name(idx) + FEATURE_SUFFIX)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, np.random.randn(n_frames + 1, 16).astype(np.float16))

    full = LibriSpeech100Dataset(
        "train-clean-100",
        kind="vocoder",
        recipe_dir=tmp_path,
        source_dir=librispeech,
        features_dir=features_dir,
        segment_frames=None,
    )
    assert len(full) == _count("train-clean-100")
    item = full[0]
    assert item["feats"].dtype == np.float32 and item["speech"].dtype == np.float32
    assert item["feats"].shape[0] * HOP == item["speech"].shape[0]

    segmented = LibriSpeech100Dataset(
        "train-clean-100",
        kind="vocoder",
        recipe_dir=tmp_path,
        source_dir=librispeech,
        features_dir=features_dir,
        segment_frames=10,
    )
    item = segmented[1]
    assert item["feats"].shape == (10, 16)
    assert item["speech"].shape == (10 * HOP,)

    # Utterance-level holdout partitions the split deterministically.
    train = LibriSpeech100Dataset(
        "train-clean-100",
        kind="vocoder",
        recipe_dir=tmp_path,
        source_dir=librispeech,
        features_dir=features_dir,
        subset="train",
        valid_ratio=0.5,
    )
    valid = LibriSpeech100Dataset(
        "train-clean-100",
        kind="vocoder",
        recipe_dir=tmp_path,
        source_dir=librispeech,
        features_dir=features_dir,
        subset="valid",
        valid_ratio=0.5,
    )
    assert len(train) + len(valid) == _count("train-clean-100")
    assert len(train) > 0 and len(valid) > 0
    assert _is_valid_utterance("x", 1.0) and not _is_valid_utterance("x", 0.0)


def test_vocoder_kind_missing_feature_file(tmp_path, librispeech):
    dataset = LibriSpeech100Dataset(
        "dev-clean",
        kind="vocoder",
        recipe_dir=tmp_path,
        source_dir=librispeech,
        features_dir=tmp_path / "nowhere",
    )
    with pytest.raises(FileNotFoundError, match="prepare_features"):
        dataset[0]


def test_conversion_kind_pairs(tmp_path, librispeech):
    dataset = LibriSpeech100Dataset(
        "dev-clean",
        kind="conversion",
        recipe_dir=tmp_path,
        source_dir=librispeech,
        num_pairs=3,
        seed=7,
        max_reference_seconds=0.8,
    )
    assert len(dataset) == 3
    item = dataset[0]
    assert set(item) == {
        "speech",
        "reference_speech",
        "target_speaker",
        "pair_id",
        "text",
    }
    source_utt, target = item["pair_id"].split("_to_")
    assert source_utt.split("-")[0] != target == item["target_speaker"]
    assert item["text"].startswith("HELLO WORLD")
    assert isinstance(item["reference_speech"], list) and item["reference_speech"]
    total = sum(len(w) for w in item["reference_speech"]) / SR
    assert total <= 0.8 or len(item["reference_speech"]) == 1

    # Same seed -> same pairs; all pairs -> one per utterance.
    again = LibriSpeech100Dataset(
        "dev-clean",
        kind="conversion",
        recipe_dir=tmp_path,
        source_dir=librispeech,
        num_pairs=3,
        seed=7,
    )
    assert [p["pair_id"] for p in (dataset[i] for i in range(3))] == [
        again[i]["pair_id"] for i in range(3)
    ]
    everything = LibriSpeech100Dataset(
        "dev-clean",
        kind="conversion",
        recipe_dir=tmp_path,
        source_dir=librispeech,
        num_pairs=None,
    )
    assert len(everything) == _count("dev-clean")

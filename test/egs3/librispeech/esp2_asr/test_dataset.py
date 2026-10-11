"""Tests for the LibriSpeech 960h recipe's Omni-IO packing and dataset."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("omniio", reason="this recipe stores audio with Omni-IO")

from egs3.librispeech.esp2_asr.dataset import (  # noqa: E402
    Dataset,
    DatasetBuilder,
    gather_training_text,
)
from egs3.librispeech.esp2_asr.dataset.builder import (  # noqa: E402
    known_splits,
    raw_archive_dir,
    raw_dirs_for,
    resolve_split_names,
    split_view_path,
)

_CHAPTERS = {
    "train-clean-100": ("103", "1240"),
    "train-clean-360": ("14", "208"),
    "train-other-500": ("20", "205"),
    "dev-clean": ("1272", "128104"),
    "dev-other": ("116", "288045"),
    "test-clean": ("1089", "134686"),
}
_UTTS_PER_CHAPTER = 3


def _write_corpus(root: Path, raw_dirs=None) -> tuple[Path, dict[str, np.ndarray]]:
    """Create a miniature LibriSpeech tree; return its root and the waveforms."""
    librispeech = root / "LibriSpeech"
    waveforms: dict[str, np.ndarray] = {}
    rng = np.random.default_rng(0)
    for raw_dir, (speaker, chapter) in _CHAPTERS.items():
        if raw_dirs is not None and raw_dir not in raw_dirs:
            continue
        chapter_dir = librispeech / raw_dir / speaker / chapter
        chapter_dir.mkdir(parents=True, exist_ok=True)
        lines = []
        for index in range(_UTTS_PER_CHAPTER):
            utt_id = f"{speaker}-{chapter}-{index:04d}"
            signal = rng.standard_normal(1600).astype(np.float32) * 0.05
            sf.write(str(chapter_dir / f"{utt_id}.flac"), signal, 16000)
            # FLAC is lossless but quantizes to 16 bit on the way in.
            waveforms[utt_id], _ = sf.read(str(chapter_dir / f"{utt_id}.flac"))
            lines.append(f"{utt_id} HELLO {raw_dir.upper()} {index}")
        (chapter_dir / f"{speaker}-{chapter}.trans.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
    return librispeech, waveforms


def test_split_definitions() -> None:
    assert "train_960" in known_splits()
    assert raw_dirs_for(resolve_split_names(["train_960"])) == [
        "train-clean-100",
        "train-clean-360",
        "train-other-500",
    ]
    # Overlapping splits list each raw directory once.
    assert raw_dirs_for(resolve_split_names(["train_100", "train_460"])) == [
        "train-clean-100",
        "train-clean-360",
    ]
    with pytest.raises(ValueError):
        resolve_split_names(["train_1000"])


def test_build_packs_every_requested_split(tmp_path: Path) -> None:
    corpus, _ = _write_corpus(tmp_path / "corpus")
    archive_root = tmp_path / "data"
    recipe_dir = tmp_path / "recipe"
    recipe_dir.mkdir()

    builder = DatasetBuilder()
    kwargs = dict(
        recipe_dir=recipe_dir,
        source_dir=corpus,
        archive_root=archive_root,
        splits=["train_960", "dev", "test_clean"],
        num_workers=0,
    )
    assert builder.is_source_prepared(**kwargs)
    assert not builder.is_built(**kwargs)

    builder.prepare_source(**kwargs)
    builder.build(**kwargs)
    assert builder.is_built(**kwargs)

    # Audio is packed once per raw directory ...
    for raw_dir in ("train-clean-100", "train-clean-360", "train-other-500"):
        archive_dir = raw_archive_dir(recipe_dir, raw_dir, archive_root)
        assert (archive_dir / "metadata.parquet").is_file()
        assert (archive_dir / "text.tsv").is_file()
        assert sorted(p.name for p in archive_dir.glob("blob_*.bin")) == ["blob_0.bin"]

    # ... and a split is a view over them, not a copy.
    for split in ("train_960", "dev", "test_clean"):
        assert split_view_path(recipe_dir, split, archive_root).is_file()
    assert not (archive_root / "archive" / "train_960").exists()

    # `dev` reuses the dev-clean and dev-other archives rather than repacking:
    # its rows name those sources.
    import pyarrow.parquet as pq

    dev = pq.read_table(split_view_path(recipe_dir, "dev", archive_root)).to_pydict()
    assert sorted(set(dev["source"])) == ["dev-clean", "dev-other"]
    assert len(dev["id"]) == 2 * _UTTS_PER_CHAPTER

    # Building is idempotent: a second build leaves the archives alone.
    archive_dir = raw_archive_dir(recipe_dir, "train-clean-100", archive_root)
    before = (archive_dir / "metadata.parquet").stat().st_mtime_ns
    builder.build(**kwargs)
    assert (archive_dir / "metadata.parquet").stat().st_mtime_ns == before


def test_dataset_returns_the_original_audio(tmp_path: Path) -> None:
    corpus, waveforms = _write_corpus(tmp_path / "corpus", raw_dirs={"test-clean"})
    archive_root = tmp_path / "data"
    recipe_dir = tmp_path / "recipe"
    recipe_dir.mkdir()

    DatasetBuilder().build(
        recipe_dir=recipe_dir,
        source_dir=corpus,
        archive_root=archive_root,
        splits=["test_clean"],
        num_workers=0,
    )

    dataset = Dataset(
        split="test_clean", recipe_dir=recipe_dir, archive_root=archive_root
    )
    assert len(dataset) == _UTTS_PER_CHAPTER

    sample = dataset[0]
    assert sorted(sample) == ["speech", "text"]
    assert sample["speech"].dtype == np.float32
    assert sample["speech"].ndim == 1
    # The blob stores the source FLAC verbatim, so the samples round-trip.
    first_utt_id = sorted(waveforms)[0]
    np.testing.assert_allclose(
        sample["speech"], waveforms[first_utt_id].astype(np.float32), atol=0
    )
    assert sample["text"].startswith("HELLO")


def test_overlapping_splits_share_one_archive(tmp_path: Path) -> None:
    """train_100 and train_960 must not pack train-clean-100 twice."""
    corpus, _ = _write_corpus(tmp_path / "corpus")
    archive_root = tmp_path / "data"
    recipe_dir = tmp_path / "recipe"
    recipe_dir.mkdir()

    builder = DatasetBuilder()
    common = dict(
        recipe_dir=recipe_dir,
        source_dir=corpus,
        archive_root=archive_root,
        num_workers=0,
    )
    builder.build(**common, splits=["train_100"])
    archive_dir = raw_archive_dir(recipe_dir, "train-clean-100", archive_root)
    packed_at = (archive_dir / "metadata.parquet").stat().st_mtime_ns

    builder.build(**common, splits=["train_960"])
    # The second split reused the existing archive and only added its view.
    assert (archive_dir / "metadata.parquet").stat().st_mtime_ns == packed_at
    assert split_view_path(recipe_dir, "train_960", archive_root).is_file()


def test_dataset_rejects_unknown_and_unpacked_splits(tmp_path: Path) -> None:
    recipe_dir = tmp_path / "recipe"
    recipe_dir.mkdir()

    with pytest.raises(ValueError):
        Dataset(split="nope", recipe_dir=recipe_dir)

    # Distributed jobs must not pack 960 h from every rank, so the default is
    # to fail when `create_dataset` has not run.
    with pytest.raises(FileNotFoundError):
        Dataset(
            split="test_clean",
            recipe_dir=recipe_dir,
            archive_root=tmp_path / "data",
        )


def test_tokenizer_text_comes_from_the_archive(tmp_path: Path) -> None:
    corpus, _ = _write_corpus(tmp_path / "corpus", raw_dirs={"train-clean-100"})
    archive_root = tmp_path / "data"
    recipe_dir = tmp_path / "recipe"
    recipe_dir.mkdir()

    DatasetBuilder().build(
        recipe_dir=recipe_dir,
        source_dir=corpus,
        archive_root=archive_root,
        splits=["train_100"],
        num_workers=0,
    )

    texts = gather_training_text(
        recipe_dir=recipe_dir, archive_root=archive_root, split="train_100"
    )
    assert len(texts) == _UTTS_PER_CHAPTER
    assert all(text.startswith("HELLO") for text in texts)

    with pytest.raises(FileNotFoundError):
        gather_training_text(
            recipe_dir=recipe_dir, archive_root=archive_root, split="train_960"
        )

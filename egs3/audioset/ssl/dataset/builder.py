"""AudioSet-2M dataset builder for BEATs pre-training."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Iterable

import soundfile as sf

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.utils.config_utils import load_config_with_defaults

logger = logging.getLogger(__name__)


def _load_builder_config() -> dict:
    config_resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(config_resource) as config_path:
        return load_config_with_defaults(str(config_path), resolve=False)["builder"]


_CFG = _load_builder_config()


@dataclass(frozen=True)
class AudioSetExample:
    """One AudioSet clip selected for the manifest."""

    source_path: Path
    audio_path: Path
    segment_seconds: float


def resolve_source_root(source_dir: str | Path | None = None) -> Path:
    """Resolve the AudioSet root from ``source_dir`` or the environment.

    Args:
        source_dir: Directory containing the segment CSVs and wav directories.
            When ``None``, the ``AUDIOSET`` environment variable is used.

    Returns:
        Path: AudioSet root directory.

    Raises:
        FileNotFoundError: If neither location is set or the directory is
            missing.
    """
    env_var = str(_CFG["source_env_var"])
    candidate = source_dir if source_dir is not None else os.environ.get(env_var)
    if not candidate:
        raise FileNotFoundError(
            f"AudioSet root is not set. Pass `create_dataset.source_dir` or set "
            f"{env_var}."
        )
    source_root = Path(candidate)
    if not source_root.is_dir():
        raise FileNotFoundError(f"AudioSet root not found: {source_root}")
    return source_root


def _iter_wav_stems(directory: Path) -> set[str]:
    """Index ``<stem>.wav`` names with one ``scandir`` pass (cheap on NFS)."""
    if not directory.is_dir():
        return set()
    with os.scandir(directory) as entries:
        return {entry.name[:-4] for entry in entries if entry.name.endswith(".wav")}


def _read_segment_list(
    csv_path: Path,
    wav_dir: Path,
    available: set[str],
    cut_dirs: tuple[Path, Path],
    cut_available: set[str],
) -> tuple[list[AudioSetExample], int]:
    """Parse one AudioSet segment CSV into examples for downloaded clips."""
    reused_cut_dir, new_cut_dir = cut_dirs
    clip_seconds = float(_CFG["clip_seconds"])
    examples: list[AudioSetExample] = []
    num_missing = 0
    with csv_path.open("r", encoding="utf-8") as reader:
        for line in reader:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            yt_id, start, end, _ = line.split(",", maxsplit=3)
            if yt_id not in available:
                num_missing += 1
                continue
            source_path = wav_dir / f"{yt_id}.wav"
            segment_seconds = float(end) - float(start)
            if segment_seconds < clip_seconds:
                cut_dir = reused_cut_dir if yt_id in cut_available else new_cut_dir
                audio_path = cut_dir / f"{yt_id}.wav"
            else:
                audio_path = source_path
            examples.append(AudioSetExample(source_path, audio_path, segment_seconds))
    return examples, num_missing


def _prepare_clip(example: AudioSetExample) -> tuple[bool, int]:
    """Cut the clip if needed and return ``(ok, num_samples)``."""
    try:
        if example.audio_path != example.source_path and not (
            example.audio_path.is_file() and example.audio_path.stat().st_size > 0
        ):
            audio, sample_rate = sf.read(str(example.source_path), dtype="float32")
            tmp_path = example.audio_path.with_suffix(".tmp.wav")
            sf.write(
                str(tmp_path),
                audio[: int(sample_rate * example.segment_seconds)],
                sample_rate,
            )
            tmp_path.replace(example.audio_path)
        info = sf.info(str(example.audio_path))
    except Exception:  # noqa: BLE001 - unreadable clips are dropped
        return False, 0
    if info.samplerate != int(_CFG["sample_rate"]) or info.channels != 1:
        return False, 0
    return True, int(info.frames)


def _write_manifest(manifest_path: Path, rows: Iterable[tuple[str, Path, int]]) -> int:
    """Atomically write ``utt_id<TAB>audio_path<TAB>num_samples`` lines."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = manifest_path.with_name(manifest_path.name + ".tmp")
    num_rows = 0
    with tmp_path.open("w", encoding="utf-8") as writer:
        for utt_id, audio_path, num_samples in rows:
            writer.write(f"{utt_id}\t{audio_path}\t{num_samples}\n")
            num_rows += 1
    tmp_path.replace(manifest_path)
    return num_rows


class AudioSetBuilder(DatasetBuilder):
    """Build AudioSet-2M manifests for BEATs pre-training.

    Port of ``egs2/audioset/ssl1/local/data_prep_as2m.py`` plus the egs2
    duration filter (``--max_wav_duration 11``):

    1. Index the downloaded clips of every segment list.
    2. Cut clips whose segment is shorter than 10 s (reusing existing cuts in
       ``<AudioSet root>/cut_wav``).
    3. Drop unreadable clips, clips that are not 16 kHz mono, and clips outside
       ``(min_wav_duration, max_wav_duration)``.
    4. Write ``<recipe_dir>/data/manifest/{train,eval}.tsv`` with
       ``utt_id<TAB>audio_path<TAB>num_samples`` rows. Utterance ids number the
       downloaded clips in segment-list order (``as2m_20k-AudioSet-<n>``), so
       ids stay stable when clips are dropped.

    The corpus itself is never modified; new cut clips go under
    ``<recipe_dir>/data/cut_wav``. Re-running is a no-op once both manifests
    exist.

    Config (``create_dataset`` block of the training config):
        recipe_dir: Recipe root directory.
        source_dir: AudioSet root. Optional when ``AUDIOSET`` is set.
    """

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Return whether the AudioSet segment lists are available."""
        try:
            source_root = resolve_source_root(source_dir)
        except FileNotFoundError:
            return False
        return all(
            (source_root / entry["csv_path"]).is_file()
            for entries in _CFG["segment_lists"].values()
            for entry in entries
        )

    def prepare_source(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        **_kwargs,
    ) -> None:
        """Validate the AudioSet root; downloading AudioSet is out of scope.

        Raises:
            FileNotFoundError: If the root or a segment list is missing.
        """
        source_root = resolve_source_root(source_dir)
        missing = [
            str(source_root / entry["csv_path"])
            for entries in _CFG["segment_lists"].values()
            for entry in entries
            if not (source_root / entry["csv_path"]).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "AudioSet segment lists are missing: " + ", ".join(missing)
            )

    def is_built(self, recipe_dir: str | Path, **_kwargs) -> bool:
        """Return whether both manifests exist."""
        data_root = Path(recipe_dir).resolve() / _CFG["data_path"]
        return all(
            (data_root / path).is_file() for path in _CFG["manifest_paths"].values()
        )

    def build(
        self,
        recipe_dir: str | Path,
        source_dir: str | Path | None = None,
        num_workers: int | None = None,
        **_kwargs,
    ) -> None:
        """Cut, validate, and filter clips, then write the manifests.

        Args:
            recipe_dir: Recipe root directory.
            source_dir: AudioSet root. Optional when ``AUDIOSET`` is set.
            num_workers: Processes used to cut and inspect clips. Defaults to
                ``builder.num_workers`` in ``dataset/config.yaml``.
            **_kwargs: Unused extra options for API compatibility.

        Raises:
            FileNotFoundError: If the AudioSet root is missing.
            RuntimeError: If a split ends up with no usable clips.
        """
        source_root = resolve_source_root(source_dir)
        data_root = Path(recipe_dir).resolve() / _CFG["data_path"]
        reused_cut_dir = source_root / _CFG["cut_wav_dir"]
        new_cut_dir = data_root / _CFG["cut_wav_dir"]
        new_cut_dir.mkdir(parents=True, exist_ok=True)
        cut_available = _iter_wav_stems(reused_cut_dir)
        sample_rate = int(_CFG["sample_rate"])
        min_samples = float(_CFG["min_wav_duration"]) * sample_rate
        max_samples = float(_CFG["max_wav_duration"]) * sample_rate
        num_workers = int(num_workers or _CFG["num_workers"])

        for split, entries in _CFG["segment_lists"].items():
            examples: list[AudioSetExample] = []
            for entry in entries:
                wav_dir = source_root / entry["wav_dir"]
                parsed, num_missing = _read_segment_list(
                    source_root / entry["csv_path"],
                    wav_dir,
                    _iter_wav_stems(wav_dir),
                    (reused_cut_dir, new_cut_dir),
                    cut_available,
                )
                logger.info(
                    "%s: %d clips found, %d not downloaded",
                    entry["csv_path"],
                    len(parsed),
                    num_missing,
                )
                examples.extend(parsed)

            with ProcessPoolExecutor(max_workers=num_workers) as pool:
                results = list(pool.map(_prepare_clip, examples, chunksize=256))
            utt_id_prefix = str(_CFG["utt_id_prefixes"][split])
            rows = [
                (f"{utt_id_prefix}-{position}", example.audio_path, num_samples)
                for position, (example, (ok, num_samples)) in enumerate(
                    zip(examples, results)
                )
                if ok and min_samples < num_samples < max_samples
            ]
            if not rows:
                raise RuntimeError(f"No usable AudioSet clips for split '{split}'.")
            num_rows = _write_manifest(data_root / _CFG["manifest_paths"][split], rows)
            logger.info(
                "%s: wrote %d clips (%d dropped as unreadable or out of range)",
                split,
                num_rows,
                len(examples) - num_rows,
            )

"""AudioSet-2M dataset builder for BEATs pre-training."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import soundfile as sf
from omegaconf import DictConfig

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.parallel.base_runner import BaseRunner
from espnet3.parallel.env_provider import EnvironmentProvider
from espnet3.parallel.parallel import set_parallel
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
            "AudioSet was not found: the AudioSet root is not set. This recipe "
            "does not download AudioSet; download it first (segment CSVs and "
            "the `*_wav` clip directories), then pass its root as "
            f"`create_dataset.source_dir` or set {env_var}."
        )
    source_root = Path(candidate)
    if not source_root.is_dir():
        raise FileNotFoundError(
            f"AudioSet was not found at {source_root}. This recipe does not "
            "download AudioSet; download it first or fix "
            f"`create_dataset.source_dir` / {env_var}."
        )
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
    """Cut the clip if needed and return ``(ok, num_samples)``.

    Only reading the source clip is allowed to fail: AudioSet downloads contain
    missing and truncated files, and those clips are dropped. Writing the cut
    clip is not guarded, so a full disk or an unwritable directory stops the
    build instead of silently shrinking the corpus.
    """
    if example.audio_path != example.source_path and not (
        example.audio_path.is_file() and example.audio_path.stat().st_size > 0
    ):
        try:
            audio, sample_rate = sf.read(str(example.source_path), dtype="float32")
        except (sf.LibsndfileError, OSError):
            return False, 0
        tmp_path = example.audio_path.with_suffix(".tmp.wav")
        sf.write(
            str(tmp_path),
            audio[: int(sample_rate * example.segment_seconds)],
            sample_rate,
        )
        tmp_path.replace(example.audio_path)
    try:
        info = sf.info(str(example.audio_path))
    except (sf.LibsndfileError, OSError):
        return False, 0
    if info.samplerate != int(_CFG["sample_rate"]) or info.channels != 1:
        return False, 0
    return True, int(info.frames)


def _write_clip_list(path: Path, examples: Iterable[AudioSetExample]) -> None:
    """Write ``source_path<TAB>audio_path<TAB>segment_seconds`` lines.

    The clip list is how the Runner workers receive the examples of one split:
    each worker reads it once in its setup, instead of receiving ~2M examples
    through the scheduler.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as writer:
        for example in examples:
            writer.write(
                f"{example.source_path}\t{example.audio_path}\t"
                f"{example.segment_seconds}\n"
            )


def _read_clip_list(path: str | Path) -> List[AudioSetExample]:
    """Read the clip list written by :func:`_write_clip_list`."""
    examples = []
    with Path(path).open("r", encoding="utf-8") as reader:
        for line in reader:
            source_path, audio_path, segment_seconds = line.rstrip("\n").split("\t")
            examples.append(
                AudioSetExample(
                    Path(source_path), Path(audio_path), float(segment_seconds)
                )
            )
    return examples


class PrepareClipProvider(EnvironmentProvider):
    """Provide the clips of one split to :class:`PrepareClipRunner` workers.

    Args:
        config: ``create_dataset`` config. Unused beyond the base class.
        clip_list_path: Clip list written by :func:`_write_clip_list`.
    """

    def __init__(self, config: DictConfig, clip_list_path: str | Path):
        """Store the clip list path; clips are loaded when the env is built."""
        super().__init__(config)
        self.clip_list_path = str(clip_list_path)

    def build_env_local(self) -> Dict[str, Any]:
        """Load the clip list on the driver for local execution."""
        return {"examples": _read_clip_list(self.clip_list_path)}

    def build_worker_setup_fn(self) -> Callable[[], Dict[str, Any]]:
        """Return a setup function that loads the clip list on each worker."""
        clip_list_path = self.clip_list_path

        def setup() -> Dict[str, Any]:
            return {"examples": _read_clip_list(clip_list_path)}

        return setup


class PrepareClipRunner(BaseRunner):
    """Cut and inspect AudioSet clips in parallel.

    Each shard writes one JSON line per clip to its ``results.jsonl``, and
    :meth:`merge` returns the records of all shards sorted by ``idx`` (the
    position of the clip in the split's clip list), whatever order the shards
    finished in.
    """

    @staticmethod
    def forward(
        idx: int | Iterable[int], examples: List[AudioSetExample], **env
    ) -> Dict[str, Any] | List[Dict[str, Any]]:
        """Prepare the clip(s) at ``idx``.

        Returns:
            ``{"idx": int, "ok": bool, "num_samples": int}`` for an int index,
            or a list of them for a batch of indices. See :func:`_prepare_clip`.
        """
        if isinstance(idx, int):
            return PrepareClipRunner._process_one(idx, examples)
        return [PrepareClipRunner._process_one(i, examples) for i in idx]

    @staticmethod
    def _process_one(idx: int, examples: List[AudioSetExample]) -> Dict[str, Any]:
        ok, num_samples = _prepare_clip(examples[idx])
        return {"idx": idx, "ok": ok, "num_samples": num_samples}

    @staticmethod
    def open_writers(shard_dir: Optional[Path], **env) -> Dict[str, Any]:
        """Open the shard-local ``results.jsonl``."""
        return {"results": (Path(shard_dir) / "results.jsonl").open("w")}

    @staticmethod
    def write_record(
        writers: Dict[str, Any], result: Any, state: Dict[str, Any], **env
    ) -> None:
        """Append one ``forward`` result (or batch of results)."""
        records = result if isinstance(result, list) else [result]
        for record in records:
            writers["results"].write(json.dumps(record) + "\n")

    @staticmethod
    def close_writers(
        writers: Dict[str, Any], state: Dict[str, Any], **env
    ) -> Dict[str, Any]:
        """Close the shard-local ``results.jsonl``."""
        writers["results"].close()
        return {}

    def merge(self, shard_dirs: List[Path]) -> List[Dict[str, Any]]:
        """Concatenate shard results in clip-list order.

        Each shard's ``results.jsonl`` holds one JSON object per clip, e.g.::

            {"idx": 0, "ok": true, "num_samples": 160000}
            {"idx": 1, "ok": false, "num_samples": 0}
        """
        records: List[Dict[str, Any]] = []
        for shard_dir in shard_dirs:
            with (Path(shard_dir) / "results.jsonl").open("r") as reader:
                records.extend(json.loads(line) for line in reader if line.strip())
        records.sort(key=lambda record: record["idx"])
        return records


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
    exist. Step 2-3 run through :class:`PrepareClipRunner`, so they use the
    ``create_dataset.parallel`` config (e.g. a local Dask cluster or a SLURM
    cluster) like the other ESPnet3 parallel stages.

    Config (``create_dataset`` block of the training config):
        recipe_dir: Recipe root directory.
        source_dir: AudioSet root. Optional when ``AUDIOSET`` is set.
        parallel: ESPnet3 parallel config used to cut and inspect clips.
            Clips are processed sequentially on the driver when omitted.

    Examples:
        .. code-block:: yaml

            create_dataset:
              recipe_dir: ${recipe_dir}
              source_dir: /path/to/audioset
              parallel:
                env: local
                n_workers: 16
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
        """Validate the AudioSet root; this recipe does not download AudioSet.

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
                "AudioSet was not found: segment lists are missing: "
                + ", ".join(missing)
                + ". This recipe does not download AudioSet; download it first."
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
        parallel: DictConfig | dict | None = None,
        **_kwargs,
    ) -> None:
        """Cut, validate, and filter clips, then write the manifests.

        Args:
            recipe_dir: Recipe root directory.
            source_dir: AudioSet root. Optional when ``AUDIOSET`` is set.
            parallel: ESPnet3 parallel config for :class:`PrepareClipRunner`.
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
        if parallel is not None:
            set_parallel(DictConfig(parallel))
        work_dir = data_root / "prepare_clips"

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

            clip_list_path = work_dir / f"{split}_clips.tsv"
            _write_clip_list(clip_list_path, examples)
            # resume=False: results depend on the clip list, which changes
            # whenever more of AudioSet is downloaded.
            runner = PrepareClipRunner(
                provider=PrepareClipProvider(DictConfig({}), clip_list_path),
                batch_size=int(_CFG["batch_size"]),
                output_dir=work_dir,
                shard_subdir=split,
                resume=False,
            )
            results = runner(range(len(examples))) if examples else []
            utt_id_prefix = str(_CFG["utt_id_prefixes"][split])
            rows = [
                (
                    f"{utt_id_prefix}-{result['idx']}",
                    examples[result["idx"]].audio_path,
                    result["num_samples"],
                )
                for result in results
                if result["ok"] and min_samples < result["num_samples"] < max_samples
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

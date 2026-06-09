"""FalAR dataset builder."""

from __future__ import annotations

import io
import json
import logging
import re
import shutil
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from datasets import load_dataset, load_dataset_builder
from datasets.download import DownloadConfig, DownloadManager
from huggingface_hub import HfApi, hf_hub_url
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.parallel.base_runner import BaseRunner
from espnet3.parallel.env_provider import EnvironmentProvider
from espnet3.parallel.parallel import set_parallel
from espnet3.utils.config_utils import load_config_with_defaults

logger = logging.getLogger(__name__)


def _load_recipe_config() -> dict:
    config_resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(config_resource) as config_path:
        return load_config_with_defaults(str(config_path), resolve=False)


_CONFIG = _load_recipe_config()
_BUILDER_CFG = _CONFIG["builder"]
_DATASET_CFG = _CONFIG["dataset"]
_HF_DATASET = str(_BUILDER_CFG["hf_dataset"])
_STATE_PATH = str(_BUILDER_CFG["prepare_state_path"])
_DEFAULT_CACHE_DIR = _BUILDER_CFG.get("cache_dir")
_ARTIFACT_ROOT = str(_BUILDER_CFG["artifact_root"])
_BUILD_STATE_PATH = str(_BUILDER_CFG["build_state_path"])
_MANIFEST_FILENAME = str(_BUILDER_CFG["manifest_filename"])
_AUDIO_BASENAME = str(_BUILDER_CFG["audio_basename"])
_TARGET_FORMAT = str(_BUILDER_CFG.get("target_format", "wav"))
_ARKIVE_APPEND_BATCH_SIZE = int(_BUILDER_CFG.get("arkive_append_batch_size", 256))
_AUDIO_COLUMN = str(_DATASET_CFG["audio_column"])
_TEXT_COLUMN = str(_DATASET_CFG["text_column"])
_ID_COLUMN = str(_DATASET_CFG["id_column"])
_TRAIN_SHARDS = [str(split) for split in _DATASET_CFG["train_shards"]]
_DEV_SPLIT = str(_DATASET_CFG["dev_split"])
_TEST_SPLIT = str(_DATASET_CFG["test_split"])
_DOWNLOAD_SPLITS = [*_TRAIN_SHARDS, _DEV_SPLIT, _TEST_SPLIT]
_PARQUET_REVISION = "refs/convert/parquet"
_PARQUET_SPLIT_RE = re.compile(r"^(?P<split>.+)-\d{5}-of-\d{5}.*\.parquet$")
_ARTIFACT_SPLITS = {
    "train": _TRAIN_SHARDS,
    "valid": [_DEV_SPLIT],
    "test": [_TEST_SPLIT],
}


def _cache_dir_as_str(cache_dir: str | Path | None) -> str | None:
    if cache_dir is None:
        return None
    return str(Path(cache_dir))


def _download_file(
    item: dict[str, str],
    cache_dir: str | None = None,
) -> dict[str, str]:
    manager = DownloadManager(
        download_config=DownloadConfig(cache_dir=cache_dir),
    )
    cached_path = manager.download(item["url"])
    return {
        "split": item["split"],
        "url": item["url"],
        "cached_path": str(cached_path),
    }


def _iter_builder_data_files(cache_dir: str | None) -> list[dict[str, str]]:
    builder = load_dataset_builder(
        _HF_DATASET,
        cache_dir=cache_dir,
    )
    data_files = getattr(builder.config, "data_files", None)
    if not data_files:
        return []

    items: list[dict[str, str]] = []
    for split_name, files in data_files.items():
        if split_name not in _DOWNLOAD_SPLITS:
            continue
        for file_url in files:
            file_url = str(file_url)
            if not file_url.endswith(".parquet"):
                continue
            items.append({"split": str(split_name), "url": file_url})
    return items


def _iter_repo_parquet_files() -> list[dict[str, str]]:
    api = HfApi()
    items: list[dict[str, str]] = []
    for repo_file in api.list_repo_files(
        repo_id=_HF_DATASET,
        repo_type="dataset",
        revision=_PARQUET_REVISION,
    ):
        if not repo_file.endswith(".parquet"):
            continue
        file_name = Path(repo_file).name
        match = _PARQUET_SPLIT_RE.match(file_name)
        if match is None:
            continue
        split_name = match.group("split")
        if split_name not in _DOWNLOAD_SPLITS:
            continue
        items.append(
            {
                "split": split_name,
                "url": hf_hub_url(
                    repo_id=_HF_DATASET,
                    filename=repo_file,
                    repo_type="dataset",
                    revision=_PARQUET_REVISION,
                ),
            }
        )
    return items


def _resolve_download_items(cache_dir: str | None) -> list[dict[str, str]]:
    items = _iter_builder_data_files(cache_dir)
    if items:
        return items
    return _iter_repo_parquet_files()


def _load_arkive_class():
    try:
        from arkive import Arkive
    except ImportError as exc:
        raise ImportError(
            "arkive is required for FalAR build(). Install it before running "
            "create_dataset."
        ) from exc
    return Arkive


def _audio_to_frames_and_rate(audio_item: Any) -> tuple[np.ndarray, int]:
    """Convert one HF audio cell to soundfile-compatible frames."""
    samples = audio_item.get_all_samples()
    sample_rate = int(getattr(samples, "sample_rate"))
    array = samples.data
    if hasattr(array, "detach"):
        array = array.detach().cpu().numpy()
    else:
        array = np.asarray(array)

    if array.ndim == 1:
        frames = array
    elif array.ndim == 2:
        # HF audio commonly returns channel-first tensors.
        if array.shape[0] <= array.shape[1]:
            frames = array.transpose(1, 0)
        else:
            frames = array
    else:
        raise ValueError(f"Unsupported audio tensor shape: {array.shape}")

    return frames.astype(np.float32), sample_rate


def _frames_to_wav_bytes(frames: np.ndarray, sample_rate: int) -> bytes:
    """Encode float32 frames as in-memory WAV bytes."""
    with io.BytesIO() as buffer:
        sf.write(buffer, frames, sample_rate, format="WAV", subtype="PCM_16")
        return buffer.getvalue()


class FalarDownloadProvider(EnvironmentProvider):
    """Provider for FalAR cache warm-up jobs."""

    def __init__(self, cache_dir: str | None) -> None:
        super().__init__(OmegaConf.create({"cache_dir": cache_dir}))

    def build_env_local(self) -> dict:
        return {"cache_dir": self.config.get("cache_dir")}

    def build_worker_setup_fn(self):
        cache_dir = self.config.get("cache_dir")

        def setup():
            return {"cache_dir": cache_dir}

        return setup


class FalarDownloadRunner(BaseRunner):
    """Runner that downloads FalAR parquet files into the Hugging Face cache."""

    @staticmethod
    def open_writers(shard_dir: Path | None, **_env) -> dict[str, object]:
        if shard_dir is None:
            raise ValueError("shard_dir must be set for FalarDownloadRunner.")
        results_path = shard_dir / "results.jsonl"
        return {
            "results": results_path.open("w", encoding="utf-8"),
        }

    @staticmethod
    def forward(item: dict[str, str], cache_dir=None, **_env):
        return _download_file(item, cache_dir=cache_dir)

    @staticmethod
    def write_record(
        writers: dict[str, object],
        result: dict[str, str],
        state: dict[str, object],
        **_env,
    ) -> None:
        del state
        results_writer = writers["results"]
        results_writer.write(json.dumps(result, ensure_ascii=False) + "\n")

    def merge(self, shard_dirs: list[Path]) -> list[dict[str, int | str]]:
        results: list[dict[str, int | str]] = []
        for shard_dir in shard_dirs:
            results_path = shard_dir / "results.jsonl"
            if not results_path.exists():
                continue
            with results_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    results.append(json.loads(line))
        return results


class FalarBuilder(DatasetBuilder):
    """Prepare FalAR arkive and manifest artifacts for recipe-local use."""

    @staticmethod
    def _state_file(recipe_dir: str | Path) -> Path:
        recipe_root = Path(recipe_dir).resolve()
        return recipe_root / _STATE_PATH

    @staticmethod
    def _build_state_file(recipe_dir: str | Path) -> Path:
        recipe_root = Path(recipe_dir).resolve()
        return recipe_root / _BUILD_STATE_PATH

    @staticmethod
    def _artifact_root(recipe_dir: str | Path) -> Path:
        recipe_root = Path(recipe_dir).resolve()
        return recipe_root / _ARTIFACT_ROOT

    @classmethod
    def _split_dir(cls, recipe_dir: str | Path, split: str) -> Path:
        return cls._artifact_root(recipe_dir) / split

    @classmethod
    def _manifest_path(cls, recipe_dir: str | Path, split: str) -> Path:
        return cls._split_dir(recipe_dir, split) / _MANIFEST_FILENAME

    @classmethod
    def _arkive_prefix(cls, recipe_dir: str | Path, split: str) -> Path:
        return cls._split_dir(recipe_dir, split) / _AUDIO_BASENAME

    @staticmethod
    def _resolve_cache_dir(
        recipe_dir: str | Path,
        cache_dir: str | Path | None,
    ) -> str | None:
        if cache_dir is not None:
            return _cache_dir_as_str(cache_dir)
        if not _DEFAULT_CACHE_DIR:
            return None
        recipe_root = Path(recipe_dir).resolve()
        return str((recipe_root / str(_DEFAULT_CACHE_DIR)).resolve())

    @staticmethod
    def _normalize_parallel_config(parallel) -> DictConfig | None:
        if parallel is None:
            return None
        if isinstance(parallel, DictConfig):
            return parallel
        return OmegaConf.create(parallel)

    @staticmethod
    def _get_num_workers(parallel) -> int:
        parallel_config = FalarBuilder._normalize_parallel_config(parallel)
        if parallel_config is None:
            return 1
        return max(1, int(parallel_config.get("n_workers", 1)))

    @staticmethod
    def _is_recipe_local_path(recipe_dir: str | Path, path: str | Path | None) -> bool:
        if path is None:
            return False
        recipe_root = Path(recipe_dir).resolve()
        target = Path(path).resolve()
        try:
            target.relative_to(recipe_root)
        except ValueError:
            return False
        return True

    @staticmethod
    def _load_dataset_split(hf_split: str, cache_dir: str | None):
        return load_dataset(
            _HF_DATASET,
            split=hf_split,
            cache_dir=cache_dir,
            download_mode="reuse_dataset_if_exists",
        )

    @staticmethod
    def _write_manifest_records(
        archive,
        pending_rows: list[dict[str, Any]],
        manifest_file,
    ) -> int:
        prev_rows = len(archive) - len(pending_rows)
        if archive.data is None:
            raise RuntimeError("Arkive append completed without metadata.")

        appended_rows = archive.data.iloc[prev_rows:]
        if len(appended_rows) != len(pending_rows):
            raise RuntimeError(
                f"Arkive metadata row count mismatch: expected {len(pending_rows)}, "
                f"got {len(appended_rows)}."
            )

        metadata_by_path: dict[str, tuple[int, dict[str, Any]]] = {}
        for archive_index, row in appended_rows.iterrows():
            metadata = row.to_dict()
            metadata_key = str(metadata["original_file_path"])
            metadata_by_path[metadata_key] = (int(archive_index), metadata)

        for pending in pending_rows:
            metadata_key = pending["metadata_key"]
            match = metadata_by_path.pop(metadata_key, None)
            if match is None:
                raise RuntimeError(
                    f"Missing Arkive metadata for appended item: {metadata_key}"
                )

            archive_index, metadata = match
            sample_rate_hz = int(metadata["sample_rate"])
            duration_seconds = metadata.get("duration_seconds")
            if duration_seconds is None:
                duration_seconds = float(metadata["length"]) / sample_rate_hz

            record = {
                "utt_id": pending["utt_id"],
                "text": pending["text"],
                "text_ctc": pending["text"],
                "text_prev": "<na>",
                "split": pending["split"],
                "arkive_path": _AUDIO_BASENAME,
                "archive_index": archive_index,
                "bin_index": int(metadata["bin_index"]),
                "start_byte_offset": int(metadata["start_byte_offset"]),
                "file_size_bytes": int(metadata["file_size_bytes"]),
                "sample_rate": sample_rate_hz,
                "channels": int(metadata["channels"]),
                "duration_seconds": float(duration_seconds),
                "format": str(metadata["format"]),
                "bit_depth": int(metadata["bit_depth"]),
            }
            manifest_file.write(json.dumps(record, ensure_ascii=False) + "\n")

        return len(pending_rows)

    def _validate_config(self) -> None:
        if not _HF_DATASET:
            raise ValueError("FalAR builder config must define builder.hf_dataset.")

    def is_source_prepared(
        self,
        recipe_dir: str | Path,
        cache_dir: str | Path | None = None,
        **_kwargs,
    ) -> bool:
        """Return whether source download can be skipped."""
        self._validate_config()
        if self.is_built(recipe_dir):
            return True

        state_path = self._state_file(recipe_dir)
        if not state_path.is_file():
            return False

        state = json.loads(state_path.read_text(encoding="utf-8"))
        expected_cache_dir = self._resolve_cache_dir(recipe_dir, cache_dir)
        return (
            state.get("hf_dataset") == _HF_DATASET
            and state.get("cache_dir") == expected_cache_dir
            and state.get("num_files", 0) > 0
        )

    def prepare_source(
        self,
        recipe_dir: str | Path,
        cache_dir: str | Path | None = None,
        parallel=None,
        **_kwargs,
    ) -> None:
        """Download FalAR parquet files, optionally in parallel."""
        self._validate_config()
        cache_dir_str = self._resolve_cache_dir(recipe_dir, cache_dir)
        items = _resolve_download_items(cache_dir_str)
        if not items:
            raise RuntimeError("No FalAR parquet files were resolved for download.")
        parallel_config = self._normalize_parallel_config(parallel)
        num_workers = self._get_num_workers(parallel_config)
        if parallel_config is not None and num_workers > 1:
            set_parallel(parallel_config)

        runner = FalarDownloadRunner(
            provider=FalarDownloadProvider(cache_dir=cache_dir_str),
            output_dir=self._state_file(recipe_dir).parent / "_prepare_source",
        )
        results = runner(items)

        state_path = self._state_file(recipe_dir)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(
            json.dumps(
                {
                    "hf_dataset": _HF_DATASET,
                    "cache_dir": cache_dir_str,
                    "num_files": len(items),
                    "splits": sorted({item["split"] for item in items}),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        logger.info(
            "Prepared FalAR source cache with %d worker(s) for %d parquet files",
            num_workers,
            len(results),
        )

    def is_built(self, recipe_dir: str | Path, **_kwargs) -> bool:
        """Check whether all arkive and manifest artifacts exist."""
        self._validate_config()
        build_state_path = self._build_state_file(recipe_dir)
        if not build_state_path.is_file():
            return False
        for split in _ARTIFACT_SPLITS:
            split_dir = self._split_dir(recipe_dir, split)
            if not split_dir.is_dir():
                return False
            if not (split_dir / f"{_AUDIO_BASENAME}.parquet").is_file():
                return False
            if not self._manifest_path(recipe_dir, split).is_file():
                return False
            if not any(split_dir.glob(f"{_AUDIO_BASENAME}*.bin")):
                return False
        return True

    def build(
        self,
        recipe_dir: str | Path,
        cache_dir: str | Path | None = None,
        parallel=None,
        **_kwargs,
    ) -> None:
        """Build arkive + jsonl artifacts and remove recipe-local HF cache."""
        self._validate_config()
        Arkive = _load_arkive_class()
        cache_dir_str = self._resolve_cache_dir(recipe_dir, cache_dir)
        parallel_config = self._normalize_parallel_config(parallel)
        arkive_num_workers = self._get_num_workers(parallel_config)
        artifact_root = self._artifact_root(recipe_dir)
        artifact_root.mkdir(parents=True, exist_ok=True)

        # Clear partial build outputs before rebuilding.
        for split in _ARTIFACT_SPLITS:
            split_dir = self._split_dir(recipe_dir, split)
            if split_dir.exists():
                shutil.rmtree(split_dir)

        build_summary: dict[str, Any] = {
            "hf_dataset": _HF_DATASET,
            "cache_dir": cache_dir_str,
            "artifact_root": str(artifact_root),
            "target_format": _TARGET_FORMAT,
            "splits": {},
        }

        for artifact_split, hf_splits in _ARTIFACT_SPLITS.items():
            split_dir = self._split_dir(recipe_dir, artifact_split)
            split_dir.mkdir(parents=True, exist_ok=True)
            arkive_prefix = self._arkive_prefix(recipe_dir, artifact_split)
            arkive_prefix.mkdir(parents=True, exist_ok=True)
            manifest_path = self._manifest_path(recipe_dir, artifact_split)
            archive = Arkive(str(arkive_prefix))
            num_rows = 0
            next_item_index = 0

            with manifest_path.open("w", encoding="utf-8") as manifest_file:
                for hf_split in hf_splits:
                    dataset = self._load_dataset_split(hf_split, cache_dir_str)
                    progress = tqdm(
                        dataset,
                        total=len(dataset),
                        desc=f"build {artifact_split}:{hf_split}",
                        unit="utt",
                    )
                    pending_rows: list[dict[str, Any]] = []
                    for item in progress:
                        utt_id = str(item[_ID_COLUMN]).strip()
                        transcript = str(item[_TEXT_COLUMN]).strip()
                        frames, sample_rate = _audio_to_frames_and_rate(
                            item[_AUDIO_COLUMN]
                        )
                        metadata_key = f"{artifact_split}:{hf_split}:{next_item_index:08d}:{utt_id}"
                        next_item_index += 1
                        pending_rows.append(
                            {
                                "utt_id": utt_id,
                                "text": transcript,
                                "split": artifact_split,
                                "metadata_key": metadata_key,
                                "arkive_item": {
                                    "bytes": _frames_to_wav_bytes(frames, sample_rate),
                                    "key": metadata_key,
                                    "format": "wav",
                                },
                            }
                        )
                        if len(pending_rows) >= _ARKIVE_APPEND_BATCH_SIZE:
                            archive.append(
                                [row["arkive_item"] for row in pending_rows],
                                target_format=_TARGET_FORMAT,
                                flush_interval=0,
                                num_workers=arkive_num_workers,
                            )
                            num_rows += self._write_manifest_records(
                                archive,
                                pending_rows,
                                manifest_file,
                            )
                            pending_rows = []

                    if pending_rows:
                        archive.append(
                            [row["arkive_item"] for row in pending_rows],
                            target_format=_TARGET_FORMAT,
                            flush_interval=0,
                            num_workers=arkive_num_workers,
                        )
                        num_rows += self._write_manifest_records(
                            archive,
                            pending_rows,
                            manifest_file,
                        )

            if num_rows == 0:
                raise RuntimeError(
                    f"No rows were built for FalAR split '{artifact_split}'."
                )

            build_summary["splits"][artifact_split] = {
                "hf_splits": hf_splits,
                "num_rows": num_rows,
                "manifest": str(manifest_path),
                "arkive_prefix": str(arkive_prefix),
            }
            logger.info(
                "Built FalAR arkive split=%s rows=%d dir=%s",
                artifact_split,
                num_rows,
                split_dir,
            )

        build_state_path = self._build_state_file(recipe_dir)
        build_state_path.parent.mkdir(parents=True, exist_ok=True)
        build_state_path.write_text(
            json.dumps(build_summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        if self._is_recipe_local_path(recipe_dir, cache_dir_str):
            shutil.rmtree(Path(cache_dir_str), ignore_errors=True)
            logger.info(
                "Removed recipe-local HF cache after arkive build: %s", cache_dir_str
            )
        else:
            logger.info(
                "Preserved HF cache because it is not recipe-local: %s", cache_dir_str
            )

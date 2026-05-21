"""FalAR dataset implementation backed by arkive manifests."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import Dataset as TorchDataset

from espnet3.utils.config_utils import load_config_with_defaults

_CONFIG_RESOURCE = resources.files(__package__).joinpath("config.yaml")
with resources.as_file(_CONFIG_RESOURCE) as _CONFIG_PATH:
    _CONFIG = load_config_with_defaults(str(_CONFIG_PATH), resolve=False)
_BUILDER_CFG = _CONFIG["builder"]
_DATASET_CFG = _CONFIG["dataset"]

_ARTIFACT_ROOT = str(_BUILDER_CFG["artifact_root"])
_MANIFEST_FILENAME = str(_BUILDER_CFG["manifest_filename"])
_LANGUAGE_PREFIX = str(_DATASET_CFG["language_prefix"])
_TASK_PREFIX = str(_DATASET_CFG["task_prefix"])
_NOTIMESTAMPS_SYMBOL = str(_DATASET_CFG["notimestamps_symbol"])
_TRAIN_SHARDS = {str(split) for split in _DATASET_CFG["train_shards"]}
_DEV_SPLIT = str(_DATASET_CFG["dev_split"])
_TEST_SPLIT = str(_DATASET_CFG["test_split"])
_SPLIT_ALIASES = {
    str(alias): str(target)
    for alias, target in _DATASET_CFG.get("split_aliases", {}).items()
}


def _load_arkive_class():
    try:
        from arkive import Arkive
    except ImportError as exc:
        raise ImportError(
            "arkive is required for FalarDataset. Install dependencies before "
            "loading the dataset."
        ) from exc
    return Arkive


def _resolve_artifact_split(split: str | None) -> str:
    """Resolve user-facing split names to built artifact split directories."""
    split_name = "train" if split is None else str(split)
    split_name = _SPLIT_ALIASES.get(split_name, split_name)
    if split_name == "train" or split_name in _TRAIN_SHARDS:
        return "train"
    if split_name == _DEV_SPLIT:
        return "valid"
    if split_name == _TEST_SPLIT:
        return "test"
    if split_name in {"valid", "validation", "test"}:
        return split_name if split_name != "validation" else "valid"
    raise ValueError(f"Unknown FalAR split '{split_name}'.")


def _read_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    if not entries:
        raise RuntimeError(f"Manifest is empty: {manifest_path}")
    return entries


class FalarDataset(TorchDataset):
    """FalAR ASR dataset backed by built arkive manifests.

    Args:
        split: Recipe split name. ``train`` resolves to the built training
            archive. ``valid`` and ``validation`` resolve to the validation
            archive. ``test`` resolves to the test archive.
        artifact_dir: Root directory that contains the built FalAR split
            directories. This should usually be ``${recipe_dir}/data/falar_arkive``.
        ratio: Fraction of the manifest to keep from the front.

    Returns:
        Sample dictionaries with ``speech``, ``text``, ``text_ctc``, and
        ``text_prev`` keys.
    """

    def __init__(
        self,
        split: str | None = "train",
        artifact_dir: str | Path | None = None,
        ratio: float = 1.0,
        cache_dir: str | Path | None = None,
    ) -> None:
        del cache_dir
        if not (0 < ratio <= 1.0):
            raise ValueError("ratio must be in the range (0, 1].")

        artifact_split = _resolve_artifact_split(split)
        artifact_root = (
            Path(artifact_dir)
            if artifact_dir is not None
            else Path(_ARTIFACT_ROOT)
        )
        self.split = artifact_split
        self.split_dir = artifact_root / artifact_split
        self.manifest_path = self.split_dir / _MANIFEST_FILENAME
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        self.entries = _read_manifest(self.manifest_path)
        if ratio < 1.0:
            keep = max(1, int(len(self.entries) * ratio))
            self.entries = self.entries[:keep]

        self._arkive_cls = _load_arkive_class()
        self._archive_cache: dict[Path, Any] = {}

    def __len__(self) -> int:
        return len(self.entries)

    def _get_archive(self, arkive_path: str) -> Any:
        prefix = (self.split_dir / arkive_path).resolve()
        archive = self._archive_cache.get(prefix)
        if archive is None:
            archive = self._arkive_cls(str(prefix))
            self._archive_cache[prefix] = archive
        return archive

    @staticmethod
    def _to_speech_array(array: Any) -> np.ndarray:
        if hasattr(array, "detach"):
            array = array.detach().cpu().numpy()
        arr = np.asarray(array, dtype=np.float32)
        if arr.ndim == 2:
            if arr.shape[1] == 1:
                arr = arr[:, 0]
            elif arr.shape[0] == 1:
                arr = arr[0]
            else:
                arr = arr[:, 0]
        return arr.astype(np.float32, copy=False)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        entry = self.entries[int(idx)]
        archive = self._get_archive(str(entry["arkive_path"]))
        audio = archive.extract_file(index=int(entry["archive_index"]))
        speech = self._to_speech_array(audio.array)
        transcript = str(entry["text_ctc"]).strip()
        text = (
            f"{_LANGUAGE_PREFIX}{_TASK_PREFIX}"
            f"{_NOTIMESTAMPS_SYMBOL} {transcript}"
        )

        return {
            "speech": speech,
            "text": text,
            "text_ctc": transcript,
            "text_prev": str(entry.get("text_prev", "<na>")),
        }

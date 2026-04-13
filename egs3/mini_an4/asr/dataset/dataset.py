"""Mini AN4 dataset implementation backed by TSV manifests."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from torch.utils.data import Dataset as TorchDataset

from egs3.mini_an4.asr.dataset.builder import MiniAn4Builder
from espnet3.utils.config_utils import load_config_with_defaults

# ---------------------------------------------------------------------------
# Module-level config loading
# ---------------------------------------------------------------------------

_CONFIG_RESOURCE = resources.files(__package__).joinpath("config.yaml")
with resources.as_file(_CONFIG_RESOURCE) as _CONFIG_PATH:
    _CONFIG = load_config_with_defaults(str(_CONFIG_PATH), resolve=False)
_DATASET_CFG = _CONFIG["dataset"]
_BUILDER_CFG = _CONFIG["builder"]

_SPLIT_MANIFEST_PATHS: dict[str, str] = {
    str(split): str(relpath)
    for split, relpath in _DATASET_CFG["split_manifest_paths"].items()
}


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ManifestEntry:
    """One manifest row: utterance id, wav path, and transcript text."""

    utt_id: str
    wav_path: Path
    text: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_manifest(manifest_path: Path) -> list[ManifestEntry]:
    """Read ``utt_id<TAB>wav_path<TAB>text`` lines as manifest entries."""
    entries: list[ManifestEntry] = []

    with manifest_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            utt_id, wav_path, text = line.split("\t", maxsplit=2)
            entries.append(
                ManifestEntry(
                    utt_id=utt_id,
                    wav_path=Path(wav_path),
                    text=text,
                )
            )

    if not entries:
        raise RuntimeError(f"Manifest is empty: {manifest_path}")

    return entries


# ---------------------------------------------------------------------------
# Public dataset class
# ---------------------------------------------------------------------------


class MiniAn4Dataset(TorchDataset):
    """Mini AN4 dataset that returns ``{\"speech\", \"text\"}`` samples."""

    def __init__(self, split: str, recipe_dir: str | Path | None = None) -> None:
        self.split = split
        recipe_root = (
            Path(recipe_dir).resolve()
            if recipe_dir is not None
            else Path(__file__).resolve().parents[1]
        )
        self.dataset_dir = recipe_root / _BUILDER_CFG["data_path"]

        builder = MiniAn4Builder()
        if not builder.is_source_prepared(recipe_dir=recipe_root):
            builder.prepare_source(recipe_dir=recipe_root)
        if not builder.is_built(recipe_dir=recipe_root):
            builder.build(recipe_dir=recipe_root)

        if split not in _SPLIT_MANIFEST_PATHS:
            known = ", ".join(sorted(_SPLIT_MANIFEST_PATHS))
            raise ValueError(f"Unknown split '{split}'. Expected one of: {known}")

        manifest_path = self.dataset_dir / _SPLIT_MANIFEST_PATHS[split]
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        self._entries = _read_manifest(manifest_path)

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        entry = self._entries[int(idx)]
        array, _sr = sf.read(str(entry.wav_path))
        return {
            "speech": np.asarray(array, dtype=np.float32),
            "text": entry.text,
        }

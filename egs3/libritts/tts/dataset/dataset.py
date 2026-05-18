"""LibriTTS dataset backed by recipe TSV manifests."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from torch.utils.data import Dataset as TorchDataset

from egs3.libritts.tts.dataset.builder import LibriTTSBuilder
from espnet3.utils.config_utils import load_config_with_defaults

_CONFIG_RESOURCE = resources.files(__package__).joinpath("config.yaml")
with resources.as_file(_CONFIG_RESOURCE) as _CONFIG_PATH:
    _CONFIG = load_config_with_defaults(str(_CONFIG_PATH), resolve=False)
_DATASET_CFG = _CONFIG["dataset"]
_BUILDER_CFG = _CONFIG["builder"]

_SPLIT_MANIFEST_PATHS: dict[str, str] = {
    str(split): str(relpath)
    for split, relpath in _DATASET_CFG["split_manifest_paths"].items()
}


@dataclass(frozen=True)
class ManifestEntry:
    utt_id: str
    wav_path: Path
    text: str
    sid: int


def _read_manifest(path: Path) -> list[ManifestEntry]:
    entries: list[ManifestEntry] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            utt_id, wav_path, text, sid = line.split("\t", maxsplit=3)
            entries.append(
                ManifestEntry(
                    utt_id=utt_id,
                    wav_path=Path(wav_path),
                    text=text,
                    sid=int(sid),
                )
            )
    if not entries:
        raise RuntimeError(f"Manifest is empty: {path}")
    return entries


class LibriTTSDataset(TorchDataset):
    """LibriTTS dataset returning text/speech/speaker-id samples."""

    def __init__(
        self,
        split: str,
        recipe_dir: str | Path | None = None,
        load_speech: bool = True,
    ) -> None:
        self.split = split
        self.load_speech = load_speech
        recipe_root = (
            Path(recipe_dir).resolve()
            if recipe_dir is not None
            else Path(__file__).resolve().parents[1]
        )
        self.data_dir = recipe_root / _BUILDER_CFG["data_path"]

        builder = LibriTTSBuilder()
        if not builder.is_built(recipe_dir=recipe_root):
            raise RuntimeError(
                "Dataset is not built yet. Run create_dataset stage first."
            )

        if split not in _SPLIT_MANIFEST_PATHS:
            raise ValueError(
                f"Unknown split '{split}'. Expected one of {sorted(_SPLIT_MANIFEST_PATHS)}"
            )
        manifest_path = self.data_dir / _SPLIT_MANIFEST_PATHS[split]
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        self._entries = _read_manifest(manifest_path)

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        entry = self._entries[int(idx)]
        sample: dict[str, Any] = {
            "utt_id": entry.utt_id,
            "text": entry.text,
            "sids": np.asarray([entry.sid], dtype=np.int64),
        }
        if self.load_speech:
            speech, _ = sf.read(str(entry.wav_path))
            sample["speech"] = np.asarray(speech, dtype=np.float32)
        return sample

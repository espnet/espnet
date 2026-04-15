"""Dataset loader for Mini AN4 TTS manifests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf
from torch.utils.data import Dataset as TorchDataset


@dataclass(frozen=True)
class ManifestEntry:
    utt_id: str
    wav_path: Path
    text: str


def _read_manifest(manifest_path: Path) -> List[ManifestEntry]:
    entries: List[ManifestEntry] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
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


class MiniAN4TTSDataset(TorchDataset):
    def __init__(
        self,
        manifest_path: str | Path,
        include_utt_id: bool = False,
    ) -> None:
        manifest_path = Path(manifest_path)
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        self._entries = _read_manifest(manifest_path)
        self._include_utt_id = include_utt_id

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, idx: int):
        entry = self._entries[int(idx)]
        array, _sr = sf.read(str(entry.wav_path))
        sample = {
            "speech": np.asarray(array, dtype=np.float32),
            "text": entry.text,
        }
        if self._include_utt_id:
            sample["utt_id"] = entry.utt_id
        return sample

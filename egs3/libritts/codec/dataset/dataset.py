"""LibriTTS codec dataset backed by recipe TSV manifests.

Audio-only: returns ``{"audio": np.float32 waveform}`` samples, matching
``CommonPreprocessor(speech_name="audio")`` from
``espnet2.tasks.gan_codec.GANCodecTask.build_preprocess_fn``.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from torch.utils.data import Dataset as TorchDataset

from egs3.libritts.codec.dataset.builder import LibriTTSBuilder
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
    """One manifest row: utterance id, wav path, transcript, and speaker id."""

    utt_id: str
    wav_path: Path
    text: str
    sid: int

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_manifest(path: Path) -> list[ManifestEntry]:
    """Read ``utt_id<TAB>wav_path<TAB>text<TAB>sid`` lines as manifest entries."""
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

# ---------------------------------------------------------------------------
# Public dataset class
# ---------------------------------------------------------------------------


class LibriTTSCodecDataset(TorchDataset):
    """LibriTTS dataset returning audio-only samples for codec training."""

    def __init__(
        self,
        split: str,
        recipe_dir: str | Path | None = None,
        manifest_path: str | Path | None = None,
        inference: bool = False,
    ) -> None:
        self.split = split
        self.inference = inference
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

        if manifest_path is not None:
            resolved_manifest = Path(manifest_path)
            if not resolved_manifest.is_absolute():
                resolved_manifest = (recipe_root / resolved_manifest).resolve()
        else:
            if split not in _SPLIT_MANIFEST_PATHS:
                raise ValueError(
                    f"Unknown split '{split}'. Expected one of "
                    f"{sorted(_SPLIT_MANIFEST_PATHS)}"
                )
            resolved_manifest = self.data_dir / _SPLIT_MANIFEST_PATHS[split]
        if not resolved_manifest.is_file():
            raise FileNotFoundError(f"Manifest not found: {resolved_manifest}")
        self.manifest_path = resolved_manifest
        self._entries = _read_manifest(resolved_manifest)
        self._by_utt_id = {entry.utt_id: entry for entry in self._entries}

    def __len__(self) -> int:
        return len(self._entries)

    def keys(self) -> list[str]:
        """Return utterance IDs in manifest order.

        ESPnet's chunk/sequence iterators sample batches of utterance-ID
        keys (from `UnsortedBatchSampler` reading the manifest as a key
        file) and index datasets with those keys directly, so
        `__getitem__` must accept both a positional int (plain
        `torch.utils.data.DataLoader` usage) and a utt_id string (ESPnet
        iterator usage).
        """
        return [entry.utt_id for entry in self._entries]

    def __getitem__(self, idx: int | str) -> dict[str, Any]:
        if isinstance(idx, str):
            entry = self._by_utt_id[idx]
        else:
            entry = self._entries[int(idx)]
        audio, _ = sf.read(str(entry.wav_path))
        sample: dict[str, Any] = {"audio": np.asarray(audio, dtype=np.float32)}
        if self.inference:
            sample.update(
                {
                    "utt_id": np.asarray(entry.utt_id),
                    "wav_path": str(entry.wav_path),
                }
            )
        return sample

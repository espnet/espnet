"""LibriTTS dataset backed by recipe TSV manifests."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
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
    """LibriTTS dataset returning text/speech/spembs samples.

    The output keys match what VITS via ``GANTTSTask`` consumes:
      - ``text``  : raw transcript string (tokenized later by ``CommonPreprocessor``)
      - ``speech``: float32 waveform
      - ``spembs``: float32 speaker embedding (x-vector) loaded from a ``.pt`` file
    """

    def __init__(
        self,
        split: str,
        recipe_dir: str | Path | None = None,
        manifest_path: str | Path | None = None,
        load_speech: bool = True,
        load_xvector: bool = True,
        xvector_dir: str | Path | None = None,
    ) -> None:
        self.split = split
        self.load_speech = load_speech
        self.load_xvector = load_xvector
        recipe_root = (
            Path(recipe_dir).resolve()
            if recipe_dir is not None
            else Path(__file__).resolve().parents[1]
        )
        self.data_dir = recipe_root / _BUILDER_CFG["data_path"]

        if self.load_xvector:
            if xvector_dir is None:
                raise ValueError(
                    "xvector_dir must be supplied when load_xvector is True. "
                    "Pass it via data_src_args in training.yaml, e.g. "
                    "xvector_dir: ${xvector.save_path}/${xvector.spk_embed_tag}_train"
                )
            self.xvector_dir = Path(xvector_dir)
            if not self.xvector_dir.is_dir():
                raise FileNotFoundError(
                    f"xvector_dir does not exist: {self.xvector_dir}. "
                    "Run compute_xvectors stage first."
                )
        else:
            self.xvector_dir = None

        builder = LibriTTSBuilder()
        if not builder.is_built(recipe_dir=recipe_root):
            raise RuntimeError(
                "Dataset is not built yet. Run create_dataset stage first."
            )

        # Caller-supplied manifest_path wins. Otherwise fall back to the
        # split-keyed default from dataset/config.yaml (unfiltered manifest).
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

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        entry = self._entries[int(idx)]
        sample: dict[str, Any] = {"text": entry.text}
        if self.load_speech:
            speech, _ = sf.read(str(entry.wav_path))
            sample["speech"] = np.asarray(speech, dtype=np.float32)
        if self.load_xvector:
            pt_path = self.xvector_dir / f"{entry.utt_id}.pt"
            if not pt_path.is_file():
                raise FileNotFoundError(f"X-vector missing: {pt_path}")
            spembs = torch.load(str(pt_path), map_location="cpu")
            if isinstance(spembs, torch.Tensor):
                spembs = spembs.numpy()
            sample["spembs"] = np.asarray(spembs, dtype=np.float32).squeeze()
        return sample

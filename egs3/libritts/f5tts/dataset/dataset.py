"""LibriTTS dataset backed by recipe TSV manifests."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset as TorchDataset

from egs3.libritts.f5tts.dataset.builder import LibriTTSBuilder
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
    """LibriTTS training/validation dataset returning text/speech samples.

    The output keys are:
      - ``text``  : raw transcript string (tokenized later by ``CommonPreprocessor``)
      - ``speech``: float32 waveform

    Evaluation does not go through this class: ``conf/inference.yaml`` reads
    the LibriSpeech-PC manifest via ``dataset/librispeech_pc.py``, whose rows
    pin one prompt per target.

    The dataset consumes the following arguments during initialization:
        - ``split``: A string key for the dataset split
        - ``recipe_dir``: Optional path to the recipe root, used to resolve the default
        - ``manifest_path``: Optional path to the manifest TSV file. If not
            supplied, the dataset will look for the default manifest path for
            the given split in the recipe's data directory.
        - ``load_speech``: Whether to load the speech waveform from disk. If False
            the sample will not include the "speech" key. Default: True.
        - ``fs``: Optional target sampling rate for the speech waveform. If supplied,
            the waveform will be resampled to this rate after loading.
            Default: None (no resampling).
    """

    def __init__(
        self,
        split: str,
        recipe_dir: str | Path | None = None,
        manifest_path: str | Path | None = None,
        load_speech: bool = True,
        fs: int | None = None,
    ) -> None:
        self.split = split
        self.load_speech = load_speech
        self.fs = fs
        recipe_root = (
            Path(recipe_dir).resolve()
            if recipe_dir is not None
            else Path(__file__).resolve().parents[1]
        )
        self.data_dir = recipe_root / _BUILDER_CFG["data_path"]

        builder = LibriTTSBuilder()
        # Guard on the LibriTTS manifests only, not on builder.is_built(), which
        # also requires the LibriSpeech-PC eval manifest. This dataset never
        # reads that file, and coupling to it would stop training on any
        # checkout whose LibriTTS manifests were built before the eval manifest
        # became part of create_dataset.
        if not builder.is_libritts_built(recipe_dir=recipe_root):
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
        sample: dict[str, Any] = {
            "text": entry.text,
        }
        if self.load_speech:
            speech, speech_fs = sf.read(str(entry.wav_path))
            sample["speech"] = np.asarray(speech, dtype=np.float32)
            if self.fs is not None and speech_fs != self.fs:
                # Resample if a target sampling rate is specified and
                # different from the original.
                sample["speech"] = (
                    torchaudio.functional.resample(
                        torch.from_numpy(sample["speech"]),
                        orig_freq=speech_fs,
                        new_freq=self.fs,
                    )
                    .numpy()
                    .astype(np.float32)
                )
        return sample

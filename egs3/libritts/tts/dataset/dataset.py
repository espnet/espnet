"""LibriTTS dataset backed by recipe TSV manifests."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio
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


def _safe_duration(wav_path: Path) -> float:
    """Audio duration in seconds (header read); +inf if unreadable.

    Returning +inf means an unreadable file is excluded by a finite
    ``[ref_min_sec, ref_max_sec]`` reference-duration filter.
    """
    try:
        return float(sf.info(str(wav_path)).duration)
    except Exception:
        return float("inf")


class LibriTTSDataset(TorchDataset):
    """LibriTTS dataset returning text/speech samples.

    The output keys are:
      - ``text``  : raw transcript string (tokenized later by ``CommonPreprocessor``)
      - ``speech``: float32 waveform

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
        - ``inference``: If True, the dataset will include additional metadata in each
            sample for inference purposes (utt_id, wav_path, raw_text). Default: False.
        - ``ref_mode``: If not None, include a reference utterance in each sample
            for zero-shot voice cloning inference. Must be one of "same_speaker" or
            "cross_speaker". Default: None (no reference).
        - ``ref_seed``: Random seed for deterministic reference selection
            when ``ref_mode`` is not None. Default: 0.
    """

    def __init__(
        self,
        split: str,
        recipe_dir: str | Path | None = None,
        manifest_path: str | Path | None = None,
        load_speech: bool = True,
        fs: int | None = None,
        inference: bool = False,
        ref_mode: str | None = None,
        ref_seed: int = 0,
        ref_max_sec: float | None = None,
        ref_min_sec: float = 0.0,
    ) -> None:
        self.split = split
        self.load_speech = load_speech
        self.inference = inference
        self.fs = fs
        if ref_mode not in (None, "same_speaker", "cross_speaker"):
            raise ValueError(
                "ref_mode must be None, 'same_speaker', or 'cross_speaker', "
                f"got {ref_mode!r}."
            )
        self.ref_mode = ref_mode
        self.ref_seed = ref_seed
        self.ref_max_sec = ref_max_sec
        self.ref_min_sec = ref_min_sec
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

        # Precompute a deterministic reference utterance per entry for zero-shot
        # voice-cloning inference (F5-TTS). The reference is drawn from the same
        # split: a different utterance of the same speaker ("same_speaker") or an
        # utterance of a different speaker ("cross_speaker"). F5 needs the
        # reference transcript, so __getitem__ returns ref_speech + ref_text.
        self._ref_idx: list[int] | None = None
        if self.ref_mode is not None:
            self._ref_idx = self._build_ref_index()

    def __len__(self) -> int:
        return len(self._entries)

    def _build_ref_index(self) -> list[int]:
        """Pick one reference entry index per target, deterministically.

        With ``ref_max_sec`` set, references are restricted to whole utterances
        whose audio duration is within ``[ref_min_sec, ref_max_sec]``. F5 needs
        ``ref_text`` to match ``ref_speech``, so we select short *whole*
        utterances (and use their exact transcript) rather than clip a long one —
        clipping the audio would desync the transcript. Keeping the prompt short
        also stops it from dominating short targets (F5's high mask ratio).
        """
        rng = random.Random(self.ref_seed)
        n = len(self._entries)

        if self.ref_max_sec is not None:
            allowed = [
                idx
                for idx in range(n)
                if self.ref_min_sec
                <= _safe_duration(self._entries[idx].wav_path)
                <= self.ref_max_sec
            ]
        else:
            allowed = list(range(n))

        by_spk: dict[int, list[int]] = defaultdict(list)
        for idx in allowed:
            by_spk[self._entries[idx].sid].append(idx)
        speakers = list(by_spk.keys())

        ref_idx: list[int] = []
        for i, entry in enumerate(self._entries):
            choice: int | None = None
            if self.ref_mode == "same_speaker":
                cands = [j for j in by_spk.get(entry.sid, []) if j != i]
                if cands:
                    choice = rng.choice(cands)
            else:  # cross_speaker
                others = [s for s in speakers if s != entry.sid]
                if others:
                    choice = rng.choice(by_spk[rng.choice(others)])
            if choice is None:
                # No in-range candidate (e.g. duration filter too strict): fall
                # back to any other utterance so inference still runs.
                pool = [j for j in allowed if j != i] or [j for j in range(n) if j != i]
                choice = rng.choice(pool) if pool else i
            ref_idx.append(choice)
        return ref_idx

    def __getitem__(self, idx: int) -> dict[str, Any]:
        entry = self._entries[int(idx)]
        # utt_id, wav_path, and raw_text are included for inference purposes only.
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
        if self._ref_idx is not None:
            ref_entry = self._entries[self._ref_idx[int(idx)]]
            ref_speech, _ = sf.read(str(ref_entry.wav_path))
            sample["ref_speech"] = np.asarray(ref_speech, dtype=np.float32)
            sample["ref_text"] = ref_entry.text
            if self.inference:
                # ref_wav_path is the speaker-similarity reference for metrics.
                sample["ref_wav_path"] = str(ref_entry.wav_path)
                sample["ref_utt_id"] = np.asarray(ref_entry.utt_id)
        if self.inference:
            # For inference, we additionally return utt_id,
            # raw_text, and wav_path for metrics calculation.
            metadata = {
                "utt_id": np.asarray(entry.utt_id),
                "wav_path": str(entry.wav_path),
                "raw_text": entry.text,
            }
            sample.update(metadata)
        return sample

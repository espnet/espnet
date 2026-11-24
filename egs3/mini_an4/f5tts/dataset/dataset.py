"""Mini AN4 dataset returning text-to-speech items for the F5-TTS system."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio

from egs3.mini_an4.esp2_asr.dataset.dataset import ManifestEntry, MiniAn4Dataset
from espnet3.contract import Field


class MiniAn4F5TTSDataset(MiniAn4Dataset):
    """Mini AN4 items for F5-TTS training and prompted inference.

    The manifests are the ASR recipe's, built by ``MiniAn4Builder`` under this
    recipe's directory. A training item is the waveform, resampled to ``fs``,
    and its transcript. In ``inference`` mode an item is a text to synthesize
    and the voice prompt F5-TTS continues from: the next utterance of the same
    split (the last one wraps around to the first), so no item is its own
    prompt. ``ref_wav_path`` is the prompt's file, which the ``measure`` stage
    reads as ``dataset:ref_wav_path``.

    Example:
        .. code-block:: python

            >>> dataset = MiniAn4F5TTSDataset(split="train", fs=24000)
            >>> sorted(dataset[0])
            ['speech', 'text']
            >>> test = MiniAn4F5TTSDataset(split="test", inference=True, fs=24000)
            >>> sorted(test[0])
            ['ref_wav_path', 'reference_speech', 'reference_text', 'text', 'utt_id']
    """

    def __init__(
        self,
        split: str,
        recipe_dir: str | Path | None = None,
        manifest_path: str | Path | None = None,
        fs: int | None = None,
        inference: bool = False,
    ) -> None:
        """Build the AN4 manifests under this recipe and load one split.

        Args:
            split: ``train``, ``valid`` or ``test``.
            recipe_dir: Recipe root; defaults to this recipe's directory.
            manifest_path: Manifest to read instead of the split's own, such
                as the one the ``remove_long_short`` stage wrote. Relative
                paths resolve against ``recipe_dir``.
            fs: Sampling rate the waveforms are resampled to; ``None`` keeps
                the recordings' own 16 kHz.
            inference: Return prompted inference items instead of training
                items.

        Raises:
            ValueError: If ``split`` is not one of the manifest splits.
            FileNotFoundError: If the AN4 archive or a manifest is missing.
            RuntimeError: If ``manifest_path`` holds no utterance.

        Example:
            .. code-block:: python

                >>> train = MiniAn4F5TTSDataset(
                ...     split="train",
                ...     manifest_path="data/manifests_filtered/train.tsv",
                ...     fs=24000,
                ... )
        """
        if recipe_dir is None:
            recipe_dir = Path(__file__).resolve().parents[1]
        super().__init__(split, recipe_dir)
        if manifest_path is not None:
            resolved_manifest = Path(manifest_path)
            if not resolved_manifest.is_absolute():
                resolved_manifest = Path(recipe_dir).resolve() / resolved_manifest
            self._entries = self._read_entries(resolved_manifest)
        self.fs = fs
        self.inference = inference
        if inference:
            self.fields = (
                Field("utt_id", "text"),
                Field("text", "text"),
                Field("reference_speech", "audio"),
                Field("reference_text", "text"),
                Field("ref_wav_path", "text"),
            )
        else:
            self.fields = (Field("speech", "audio"), Field("text", "text"))

    @staticmethod
    def _read_entries(manifest_path: Path) -> list[ManifestEntry]:
        """Read ``utt_id<TAB>wav_path<TAB>text`` rows, ignoring extra columns."""
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        entries = []
        with manifest_path.open("r", encoding="utf-8") as manifest_file:
            for line in manifest_file:
                if not line.strip():
                    continue
                utt_id, wav_path, text = line.rstrip("\n").split("\t")[:3]
                entries.append(ManifestEntry(utt_id, Path(wav_path), text))
        if not entries:
            raise RuntimeError(f"Manifest is empty: {manifest_path}")
        return entries

    def _load_waveform(self, wav_path: Path) -> np.ndarray:
        """Read one recording as mono float32, resampled to ``fs`` if set."""
        speech, sample_rate = sf.read(str(wav_path), dtype="float32")
        if speech.ndim > 1:
            speech = speech.mean(axis=1)
        if self.fs is not None and sample_rate != self.fs:
            speech = torchaudio.functional.resample(
                torch.from_numpy(speech), sample_rate, self.fs
            ).numpy()
        return np.asarray(speech, dtype=np.float32)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return one training item, or one prompted inference item.

        Args:
            idx: Positional index of the utterance.

        Returns:
            In training mode, a dict with the waveform under ``speech`` as a
            1-D float32 numpy array and its transcript under ``text``. In
            ``inference`` mode, a dict with ``utt_id`` and ``text`` of the
            utterance to synthesize, and ``reference_speech``,
            ``reference_text`` and ``ref_wav_path`` of its voice prompt.

        Raises:
            IndexError: If ``idx`` is out of range.

        Example:
            .. code-block:: python

                >>> test = MiniAn4F5TTSDataset(split="test", inference=True, fs=24000)
                >>> test[0]["utt_id"], Path(test[0]["ref_wav_path"]).name
                ('fcaw-cen8-b', 'mmxg-cen8-b.wav')
        """
        entry = self._entries[int(idx)]
        if not self.inference:
            return {
                "speech": self._load_waveform(entry.wav_path),
                "text": entry.text,
            }
        prompt = self._entries[(int(idx) + 1) % len(self._entries)]
        return {
            "utt_id": str(entry.utt_id),
            "text": entry.text,
            "reference_speech": self._load_waveform(prompt.wav_path),
            "reference_text": prompt.text,
            "ref_wav_path": str(prompt.wav_path),
        }

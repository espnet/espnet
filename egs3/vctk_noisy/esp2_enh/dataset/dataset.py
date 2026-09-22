"""VCTK-Noisy (VCTK-DEMAND) enhancement dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from torch.utils.data import Dataset as TorchDataset

from egs3.vctk_noisy.esp2_enh.dataset.builder import resolve_dataset_dir

# Same speaker hold-out as egs2/vctk_noisy/enh1/local/vctk_data_prep.sh
_VALID_SPEAKERS = frozenset({"p226", "p287"})


def _speaker_id(utt_id: str) -> str:
    return utt_id.split("_", 1)[0]


def _list_pairs(noisy_dir: Path, clean_dir: Path) -> list[tuple[str, Path, Path]]:
    """Return ``(utt_id, noisy_path, clean_path)`` sorted by utt id."""
    pairs: list[tuple[str, Path, Path]] = []
    for noisy_path in sorted(noisy_dir.rglob("*.wav")):
        utt_id = noisy_path.stem
        clean_path = clean_dir / noisy_path.name
        if not clean_path.is_file():
            # Some dumps nest speaker dirs under clean/; try relative path.
            clean_path = clean_dir / noisy_path.relative_to(noisy_dir)
        if not clean_path.is_file():
            raise FileNotFoundError(
                f"Missing clean reference for {noisy_path}: " f"expected {clean_path}"
            )
        pairs.append((utt_id, noisy_path, clean_path))
    if not pairs:
        raise RuntimeError(f"No wav files found under {noisy_dir}")
    return pairs


class VCTKNoisyDataset(TorchDataset):
    """Load noisy/clean pairs from a VCTK-DEMAND root.

    Args:
        split: ``train``, ``valid``, or ``test``.
        data_path: Root directory containing the four official wav folders.
            Defaults to the ``$VCTK_DEMAND`` environment variable named by
            ``dataset/config.yaml``.
        chunk_length: If set, randomly crop that many samples on train/valid.
            Ignored when ``inference`` is True or the utterance is shorter.
        inference: When True, always return the full utterance (no chunking).
    """

    def __init__(
        self,
        split: str,
        data_path: str | Path | None = None,
        chunk_length: int | None = None,
        inference: bool = False,
        **_kwargs,
    ) -> None:
        self.split = split
        self.data_path = resolve_dataset_dir(data_path)
        self.chunk_length = None if inference else chunk_length
        self.inference = inference

        if split == "test":
            noisy_dir = self.data_path / "noisy_testset_wav"
            clean_dir = self.data_path / "clean_testset_wav"
            pairs = _list_pairs(noisy_dir, clean_dir)
        elif split in ("train", "valid"):
            noisy_dir = self.data_path / "noisy_trainset_28spk_wav"
            clean_dir = self.data_path / "clean_trainset_28spk_wav"
            pairs = _list_pairs(noisy_dir, clean_dir)
            if split == "train":
                pairs = [p for p in pairs if _speaker_id(p[0]) not in _VALID_SPEAKERS]
            else:
                pairs = [p for p in pairs if _speaker_id(p[0]) in _VALID_SPEAKERS]
        else:
            raise ValueError(
                f"Unknown split '{split}'. " "Expected one of: train, valid, test"
            )

        if not pairs:
            raise RuntimeError(
                f"No utterances left for split '{split}' under " f"{self.data_path}"
            )
        self._pairs = pairs

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        utt_id, noisy_path, clean_path = self._pairs[int(idx)]
        speech_mix, _ = sf.read(str(noisy_path), dtype="float32", always_2d=False)
        speech_ref, _ = sf.read(str(clean_path), dtype="float32", always_2d=False)
        speech_mix = np.asarray(speech_mix, dtype=np.float32)
        speech_ref = np.asarray(speech_ref, dtype=np.float32)
        if speech_mix.ndim > 1:
            speech_mix = speech_mix[:, 0]
        if speech_ref.ndim > 1:
            speech_ref = speech_ref[:, 0]

        length = min(speech_mix.shape[0], speech_ref.shape[0])
        speech_mix = speech_mix[:length]
        speech_ref = speech_ref[:length]

        if self.chunk_length is not None and length > self.chunk_length:
            start = int(np.random.randint(0, length - self.chunk_length + 1))
            end = start + self.chunk_length
            speech_mix = speech_mix[start:end]
            speech_ref = speech_ref[start:end]

        return {
            "speech_mix": speech_mix,
            "speech_ref1": speech_ref,
        }

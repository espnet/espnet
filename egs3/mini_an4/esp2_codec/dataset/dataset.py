"""Mini AN4 dataset returning waveform-only items for the codec system."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from egs3.mini_an4.esp2_asr.dataset.dataset import MiniAn4Dataset
from espnet3.contract import Field


class MiniAn4CodecDataset(MiniAn4Dataset):
    """Mini AN4 items with the waveform under ``speech`` and no transcript.

    The manifests are the ASR recipe's, built by ``MiniAn4Builder`` under this
    recipe's directory. In ``inference`` mode each item also carries
    ``utt_id``, which heads every inference output, and ``wav_path``, the file
    the ``measure`` stage scores against (bound as ``dataset:wav_path`` in
    the template ``metrics.yaml``).

    Examples:
        >>> dataset = MiniAn4CodecDataset(split="train")
        >>> sorted(dataset[0])
        ['speech']
        >>> test = MiniAn4CodecDataset(split="test", inference=True)
        >>> sorted(test[0])
        ['speech', 'utt_id', 'wav_path']
    """

    fields = (
        Field("speech", "audio"),
        Field("utt_id", "text", optional=True),
        Field("wav_path", "text", optional=True),
    )

    def __init__(
        self,
        split: str,
        recipe_dir: str | Path | None = None,
        inference: bool = False,
    ) -> None:
        """Build the AN4 manifests under this recipe and load one split.

        Args:
            split: ``train``, ``valid`` or ``test``.
            recipe_dir: Recipe root; defaults to this recipe's directory.
            inference: Add ``utt_id`` and ``wav_path`` to each item.
        """
        if recipe_dir is None:
            recipe_dir = Path(__file__).resolve().parents[1]
        super().__init__(split, recipe_dir)
        self.inference = inference
        self._by_utt_id = {entry.utt_id: entry for entry in self._entries}

    def keys(self) -> list[str]:
        """Return the utterance ids in manifest order, for the chunk iterator."""
        return [entry.utt_id for entry in self._entries]

    def __getitem__(self, idx: int | str) -> dict[str, Any]:
        """Return the waveform of one item, plus its id and path for inference.

        Args:
            idx: A positional index, or an utterance id as :meth:`keys` lists
                them (the chunk iterator addresses items by id).
        """
        if isinstance(idx, str):
            entry = self._by_utt_id[idx]
        else:
            entry = self._entries[int(idx)]
        array, _sr = sf.read(str(entry.wav_path))
        sample: dict[str, Any] = {"speech": np.asarray(array, dtype=np.float32)}
        if self.inference:
            sample["utt_id"] = str(entry.utt_id)
            sample["wav_path"] = str(entry.wav_path)
        return sample

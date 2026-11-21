"""Mini AN4 dataset returning waveform-only items for the LACE system."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from egs3.mini_an4.esp2_asr.dataset.dataset import MiniAn4Dataset
from espnet3.contract import Field


class MiniAn4LACEDataset(MiniAn4Dataset):
    """Mini AN4 items with the waveform under ``audio`` and no transcript.

    The manifests are the ASR recipe's, built by ``MiniAn4Builder`` under this
    recipe's directory. In ``inference`` mode each item also carries
    ``utt_id``, which heads every inference output, and ``wav_path``, the file
    the ``measure`` stage scores against (bound as ``dataset:wav_path`` in
    ``conf/metrics.yaml``).

    Example:
        .. code-block:: python

            >>> dataset = MiniAn4LACEDataset(split="train")
            >>> sorted(dataset[0])
            ['audio']
            >>> test = MiniAn4LACEDataset(split="test", inference=True)
            >>> sorted(test[0])
            ['audio', 'utt_id', 'wav_path']
    """

    fields = (
        Field("audio", "audio"),
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

        Raises:
            ValueError: If ``split`` is not one of the manifest splits.
            FileNotFoundError: If the AN4 archive or the manifest is missing.

        Example:
            .. code-block:: python

                >>> train = MiniAn4LACEDataset(split="train")
                >>> test = MiniAn4LACEDataset(split="test", inference=True)
        """
        if recipe_dir is None:
            recipe_dir = Path(__file__).resolve().parents[1]
        super().__init__(split, recipe_dir)
        self.inference = inference
        self._by_utt_id = {entry.utt_id: entry for entry in self._entries}

    def keys(self) -> list[str]:
        """Return the utterance ids in manifest order, for the chunk iterator.

        Returns:
            List of utterance ids, in the same order as integer indexing.

        Example:
            .. code-block:: python

                >>> dataset = MiniAn4LACEDataset(split="test")
                >>> dataset.keys()
                ['fcaw-cen8-b', 'mmxg-cen8-b']
        """
        return [entry.utt_id for entry in self._entries]

    def __getitem__(self, idx: int | str) -> dict[str, Any]:
        """Return the waveform of one item, plus its id and path for inference.

        Args:
            idx: A positional index, or an utterance id as :meth:`keys` lists
                them (the chunk iterator addresses items by id).

        Returns:
            Dict with the waveform under ``audio`` as a 1-D float32 numpy
            array. In ``inference`` mode it also carries ``utt_id`` and
            ``wav_path``, both plain strings.

        Raises:
            KeyError: If a string ``idx`` is not a known utterance id.
            IndexError: If an integer ``idx`` is out of range.

        Example:
            .. code-block:: python

                >>> dataset = MiniAn4LACEDataset(split="test")
                >>> dataset[0]["audio"].dtype
                dtype('float32')
        """
        if isinstance(idx, str):
            entry = self._by_utt_id[idx]
        else:
            entry = self._entries[int(idx)]
        array, _sr = sf.read(str(entry.wav_path))
        sample: dict[str, Any] = {"audio": np.asarray(array, dtype=np.float32)}
        if self.inference:
            sample["utt_id"] = str(entry.utt_id)
            sample["wav_path"] = str(entry.wav_path)
        return sample

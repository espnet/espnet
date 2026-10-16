"""Mini AN4 LID dataset backed by TSV manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from egs3.mini_an4.esp2_lid.dataset.builder import resolve_manifest_path
from espnet3.api.inference import Field
from espnet3.components.data.base_dataset import BaseDataset


class MiniAn4LIDDataset(BaseDataset):
    """Mini AN4 dataset that returns ``{"speech", "lid_labels"}`` samples.

    Args:
        split: One of ``train``, ``valid``, or ``test``.
        recipe_dir: Optional recipe root. When omitted, defaults to this
            recipe's directory.

    Raises:
        ValueError: If ``split`` is unknown.
        FileNotFoundError: If the manifest for ``split`` does not exist; run
            the ``create_dataset`` stage first.

    Example:
        >>> dataset = MiniAn4LIDDataset(split="valid")
        >>> sorted(dataset[0])
        ['lid_labels', 'speech']
    """

    fields = (Field("speech", "audio"), Field("lid_labels", "text"))

    def __init__(self, split: str, recipe_dir: str | Path | None = None) -> None:
        """Read the manifest of one split."""
        recipe_root = (
            Path(recipe_dir).resolve()
            if recipe_dir is not None
            else Path(__file__).resolve().parents[1]
        )
        manifest = resolve_manifest_path(recipe_root, split)
        if not manifest.is_file():
            raise FileNotFoundError(
                f"Manifest not found: {manifest}. Run the create_dataset stage first."
            )
        self._entries = [
            line.split("\t")
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line
        ]

    def __len__(self) -> int:
        """Return the number of utterances in the split."""
        return len(self._entries)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return the waveform and language label of one utterance."""
        _utt_id, wav_path, language = self._entries[int(idx)]
        array, _sr = sf.read(wav_path)
        return {"speech": np.asarray(array, dtype=np.float32), "lid_labels": language}

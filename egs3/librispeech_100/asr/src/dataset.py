"""LibriSpeech dataset loader for ESPnet3 recipes.

This module provides :class:`LibriSpeechDataset`, a small Torch-style dataset
implementation that reads the *original* LibriSpeech directory structure on disk.

Design goals
------------
- Minimal and explicit: it only reads audio + transcript and returns them.
- Config-friendly: it is meant to be instantiated by Hydra via
  ``_target_: src.dataset.LibriSpeechDataset``.
- "Just works" defaults: when ``data_dir`` is not provided, it falls back to the
  ``LIBRISPEECH`` environment variable; the default split is ``train-clean-100``.

What this dataset returns
-------------------------
Each ``__getitem__`` returns a plain dict:

- ``speech``: ``np.float32`` waveform array (loaded from a ``.flac`` file)
- ``text``: transcript string

There is no resampling, no feature extraction, and no optional return flags.
Those should be handled elsewhere in the pipeline (frontend / preprocessor).

Expected directory layout
-------------------------
This loader expects a LibriSpeech tree like:

```
<data_dir>/LibriSpeech/
  train-clean-100/
  dev-clean/
  dev-other/
  test-clean/
  test-other/
```

For convenience, ``data_dir`` may point to either:
- the directory that *contains* ``LibriSpeech/`` (recommended), or
- the ``LibriSpeech/`` directory itself.

How split scanning works
------------------------
The split directory is scanned by walking for ``*.trans.txt`` files. Each line is
parsed as:

```
<utt_id> <word1> <word2> ...
```

and the corresponding audio is expected at ``<utt_id>.flac`` in the same folder.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import soundfile as sf
from torch.utils.data import Dataset as TorchDataset


@dataclass(frozen=True)
class LibriSpeechExample:
    """Internal index entry derived from a LibriSpeech transcript line."""

    utt_id: str
    flac_path: Path
    text: str


def _resolve_librispeech_root(data_dir: str | Path) -> Path:
    """Resolve the LibriSpeech root directory.

    Args:
        data_dir: Either the directory that contains ``LibriSpeech/`` or the
            ``LibriSpeech/`` directory itself.

    Returns:
        Path to the resolved ``LibriSpeech`` root directory.

    Raises:
        FileNotFoundError: If the LibriSpeech root cannot be found.
    """
    p = Path(data_dir)
    if (p / "LibriSpeech").is_dir():
        return p / "LibriSpeech"
    if p.name == "LibriSpeech" and p.is_dir():
        return p
    raise FileNotFoundError(
        "Could not find LibriSpeech root. Expected either:\n"
        f"  - {p}/LibriSpeech/\n"
        f"  - {p} (when it is the LibriSpeech directory itself)"
    )


def _parse_transcripts(split_dir: Path) -> List[LibriSpeechExample]:
    """Build an index for one split by reading transcript files.

    Args:
        split_dir: The on-disk directory for a split (e.g., ``.../dev-clean``).

    Returns:
        A list of :class:`LibriSpeechExample` entries.

    Raises:
        RuntimeError: If no valid transcript/audio pairs are found.
    """
    examples: List[LibriSpeechExample] = []

    for root, _dirs, files in os.walk(split_dir):
        for file_name in files:
            if not file_name.endswith(".trans.txt"):
                continue
            transcript_path = Path(root) / file_name
            with transcript_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    utt_id, *words = line.split()
                    text = " ".join(words)
                    flac_path = Path(root) / f"{utt_id}.flac"
                    if not flac_path.is_file():
                        continue
                    examples.append(
                        LibriSpeechExample(
                            utt_id=utt_id,
                            flac_path=flac_path,
                            text=text,
                        )
                    )

    if not examples:
        raise RuntimeError(
            f"No transcripts found under: {split_dir}. "
            "Check that the split is extracted and the path is correct."
        )
    return examples


class LibriSpeechDataset(TorchDataset):
    """Torch dataset that reads LibriSpeech from disk.

    Args:
        data_dir: Path to the LibriSpeech dataset root. If omitted, the dataset
            reads ``LIBRISPEECH`` from the environment.
        split: LibriSpeech split directory name (default: ``train-clean-100``).

    Raises:
        FileNotFoundError: If ``data_dir`` is missing and ``LIBRISPEECH`` is not set,
            or if the resolved split directory does not exist.
        RuntimeError: If no transcript/audio pairs are found for the split.

    Notes:
        - This dataset returns *decoded* waveforms using ``soundfile``.
        - Audio is returned as float32. Channel handling is delegated to soundfile
          (multi-channel inputs will be returned as a 2D array).
    """

    def __init__(
        self,
        data_dir: str | Path | None = None,
        split: str = "train-clean-100",
    ) -> None:
        if data_dir is None:
            data_dir = os.environ.get("LIBRISPEECH")
            if not data_dir:
                raise FileNotFoundError(
                    "LibriSpeech data_dir not provided and env var "
                    "LIBRISPEECH is not set."
                )
        self.librispeech_root = _resolve_librispeech_root(data_dir)
        self.split = str(split)

        split_dir = self.librispeech_root / self.split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        self._examples: List[LibriSpeechExample] = _parse_transcripts(split_dir)

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> Dict:
        """Load one item by index.

        Returns:
            dict with keys:
              - ``speech``: np.float32 waveform
              - ``text``: transcript string
        """
        ex = self._examples[int(idx)]
        array, _sr = sf.read(str(ex.flac_path))
        return {
            "speech": np.asarray(array, dtype=np.float32),
            "text": ex.text,
        }

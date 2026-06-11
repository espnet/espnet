"""LibriSpeech 100h dataset implementation backed by raw corpus directories."""

from __future__ import annotations

import os
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from torch.utils.data import Dataset as TorchDataset

from egs3.librispeech_100.asr.dataset.builder import (
    LibriSpeech100Builder,
    resolve_source_root,
)
from espnet3.utils.config_utils import load_config_with_defaults

_CONFIG_RESOURCE = resources.files(__package__).joinpath("config.yaml")
with resources.as_file(_CONFIG_RESOURCE) as _CONFIG_PATH:
    _CONFIG = load_config_with_defaults(str(_CONFIG_PATH), resolve=False)
_DATASET_CFG = _CONFIG["dataset"]

_KNOWN_SPLITS = {str(split) for split in _DATASET_CFG["supported_splits"]}


@dataclass(frozen=True)
class LibriSpeechExample:
    """Internal index entry derived from a LibriSpeech transcript line."""

    utt_id: str
    audio_path: Path
    text: str


def _scan_split(split_dir: Path) -> list[LibriSpeechExample]:
    """Build an index for one split by reading transcript files."""
    examples: list[LibriSpeechExample] = []

    for root, _dirs, files in os.walk(split_dir):
        root_path = Path(root)
        for file_name in files:
            if not file_name.endswith(".trans.txt"):
                continue
            transcript_path = root_path / file_name
            with transcript_path.open("r", encoding="utf-8") as fh:
                for raw_line in fh:
                    line = raw_line.strip()
                    if not line:
                        continue
                    utt_id, *words = line.split()
                    if not words:
                        continue
                    audio_path = root_path / f"{utt_id}.flac"
                    if not audio_path.is_file():
                        continue
                    examples.append(
                        LibriSpeechExample(
                            utt_id=utt_id,
                            audio_path=audio_path.resolve(),
                            text=" ".join(words),
                        )
                    )

    if not examples:
        raise RuntimeError(
            f"No transcript/audio pairs found under: {split_dir}. "
            "Check that the split is extracted and the path is correct."
        )
    return sorted(examples, key=lambda example: example.utt_id)


class LibriSpeech100Dataset(TorchDataset):
    """Torch dataset that reads LibriSpeech from the original directory layout.

    Args:
        split: LibriSpeech split directory name such as ``train-clean-100``.
        recipe_dir: Optional recipe root. When omitted, defaults to the current
            recipe directory inferred from this module.
        source_dir: Optional LibriSpeech parent/root override.

    Raises:
        ValueError: If ``split`` is unknown.
        FileNotFoundError: If the resolved source root or split directory does
            not exist.
        RuntimeError: If no transcript/audio pairs are found for the split.

    Examples:
        >>> dataset = LibriSpeech100Dataset(split="train-clean-100")
        >>> sample = dataset[0]
        >>> sorted(sample.keys())
        ['speech', 'text', 'utt_id']
    """

    def __init__(
        self,
        split: str,
        recipe_dir: str | Path | None = None,
        source_dir: str | Path | None = None,
    ) -> None:
        self.split = str(split)
        if self.split not in _KNOWN_SPLITS:
            known = ", ".join(sorted(_KNOWN_SPLITS))
            raise ValueError(f"Unknown split '{self.split}'. Expected one of: {known}")

        recipe_root = (
            Path(recipe_dir).resolve()
            if recipe_dir is not None
            else Path(__file__).resolve().parents[1]
        )

        builder = LibriSpeech100Builder()
        if not builder.is_source_prepared(
            recipe_dir=recipe_root,
            source_dir=source_dir,
        ):
            builder.prepare_source(recipe_dir=recipe_root, source_dir=source_dir)

        self.librispeech_root = resolve_source_root(
            recipe_root,
            source_dir=source_dir,
        )
        split_dir = self.librispeech_root / self.split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        self._examples = _scan_split(split_dir)

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        example = self._examples[int(idx)]
        array, _sr = sf.read(str(example.audio_path))
        return {
            "utt_id": example.utt_id,
            "speech": np.asarray(array, dtype=np.float32),
            "text": example.text,
        }

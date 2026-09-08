"""AISHELL-1 dataset implementation backed by raw corpus directories."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from torch.utils.data import Dataset as TorchDataset

from egs3.aishell.asr.dataset.builder import (
    AishellBuilder,
    resolve_source_root,
)
from espnet3.utils.config_utils import load_config_with_defaults

_CONFIG_RESOURCE = resources.files(__package__).joinpath("config.yaml")
with resources.as_file(_CONFIG_RESOURCE) as _CONFIG_PATH:
    _CONFIG = load_config_with_defaults(str(_CONFIG_PATH), resolve=False)
_DATASET_CFG = _CONFIG["dataset"]

_KNOWN_SPLITS = {str(split) for split in _DATASET_CFG["supported_splits"]}


@dataclass(frozen=True)
class AishellExample:
    """Internal index entry derived from an AISHELL-1 transcript line."""

    utt_id: str
    audio_path: Path
    text: str


def _load_transcripts(transcript_path: Path) -> dict[str, str]:
    """Load the utterance-to-transcription mapping from AISHELL-1."""
    transcripts = {}
    for raw_line in transcript_path.read_text(encoding="utf-8").splitlines():
        utt_id, separator, text = raw_line.strip().partition(" ")
        if utt_id and separator and text:
            transcripts[utt_id] = text.replace(" ", "")
    return transcripts


def _scan_split(split_dir: Path, transcripts: dict[str, str]) -> list[AishellExample]:
    """Build an index for one AISHELL-1 audio split."""
    examples: list[AishellExample] = []
    missing_transcripts = []
    for audio_path in sorted(split_dir.rglob("*.wav")):
        utt_id = audio_path.stem
        text = transcripts.get(utt_id)
        if text is None:
            missing_transcripts.append(utt_id)
            continue
        examples.append(
            AishellExample(
                utt_id=utt_id,
                audio_path=audio_path.resolve(),
                text=text,
            )
        )

    if missing_transcripts:
        raise RuntimeError(
            f"{len(missing_transcripts)} AISHELL-1 audio files have no transcript "
            f"under {split_dir}; first ID: {missing_transcripts[0]}"
        )

    if not examples:
        raise RuntimeError(
            f"No transcript/audio pairs found under: {split_dir}. "
            "Check that the split is extracted and the path is correct."
        )
    return sorted(examples, key=lambda example: example.utt_id)


class AishellDataset(TorchDataset):
    """Torch dataset that reads AISHELL-1 from the original directory layout.

    Args:
        split: AISHELL-1 split name: ``train``, ``dev``, or ``test``.
        recipe_dir: Optional recipe root. When omitted, defaults to the current
            recipe directory inferred from this module.
        source_dir: Optional AISHELL-1 parent/root override.

    Raises:
        ValueError: If ``split`` is unknown.
        FileNotFoundError: If the resolved source root or split directory does
            not exist.
        RuntimeError: If no transcript/audio pairs are found for the split.

    Examples:
        >>> dataset = AishellDataset(split="train")
        >>> sample = dataset[0]
        >>> sorted(sample.keys())
        ['speech', 'text']
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

        builder = AishellBuilder()
        if not builder.is_source_prepared(
            recipe_dir=recipe_root,
            source_dir=source_dir,
        ):
            builder.prepare_source(recipe_dir=recipe_root, source_dir=source_dir)

        self.aishell_root = resolve_source_root(
            recipe_root,
            source_dir=source_dir,
        )
        split_dir = self.aishell_root / "wav" / self.split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        transcripts = _load_transcripts(
            self.aishell_root / "transcript" / "aishell_transcript_v0.8.txt"
        )
        self._examples = _scan_split(split_dir, transcripts)

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        example = self._examples[int(idx)]
        array, _sr = sf.read(str(example.audio_path), dtype="float32")
        if array.ndim == 2:
            array = array.mean(axis=1)
        return {
            "speech": np.asarray(array, dtype=np.float32),
            "text": example.text,
        }

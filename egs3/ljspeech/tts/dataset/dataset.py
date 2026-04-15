"""LJSpeech TTS dataset implementation backed by the raw corpus files."""

from __future__ import annotations

import os
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from torch.utils.data import Dataset as TorchDataset

from espnet3.utils.config_utils import load_config_with_defaults

_CONFIG_RESOURCE = resources.files(__package__).joinpath("config.yaml")
with resources.as_file(_CONFIG_RESOURCE) as _CONFIG_PATH:
    _CONFIG = load_config_with_defaults(str(_CONFIG_PATH), resolve=False)
_DATASET_CFG = _CONFIG["dataset"]
_BUILDER_CFG = _CONFIG["builder"]


@dataclass(frozen=True)
class CorpusEntry:
    """One LJSpeech corpus row: utterance id, wav path, and normalized text."""

    utt_id: str
    wav_path: Path
    text: str


def _read_metadata(corpus_root: Path) -> list[CorpusEntry]:
    """Read corpus metadata and resolve it to wav/text entries."""
    entries: list[CorpusEntry] = []
    metadata_path = corpus_root / "metadata.csv"
    with metadata_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            utt_id, _unused, text = line.split("|", maxsplit=2)
            entries.append(
                CorpusEntry(
                    utt_id=utt_id,
                    wav_path=corpus_root / "wavs" / f"{utt_id}.wav",
                    text=text,
                )
            )

    if not entries:
        raise RuntimeError(f"LJSpeech metadata is empty: {metadata_path}")

    return entries


def _resolve_corpus_root(
    recipe_root: Path,
    corpus_root: str | Path | None,
) -> Path:
    """Resolve the prepared LJSpeech corpus root for dataset loading."""
    candidates: list[Path] = []
    if corpus_root is not None:
        candidates.append(Path(corpus_root))

    env_root = os.environ.get("LJSPEECH")
    if env_root:
        candidates.append(Path(env_root))

    candidates.extend(
        [
            recipe_root / _BUILDER_CFG["source_path"],
            recipe_root / "downloads" / "LJSpeech-1.1",
            Path("downloads") / "LJSpeech-1.1",
        ]
    )

    for candidate in candidates:
        resolved = candidate.expanduser()
        if resolved.name == "LJSpeech-1.1" and (resolved / "metadata.csv").is_file():
            return resolved
        nested = resolved / "LJSpeech-1.1"
        if (nested / "metadata.csv").is_file():
            return nested

    raise FileNotFoundError(
        "LJSpeech corpus not found. Run the create_dataset stage first, "
        "set LJSPEECH, or pass data_src_args.corpus_root."
    )


def _select_split(entries: list[CorpusEntry], split: str) -> list[CorpusEntry]:
    """Select the configured split using the same ordering as egs2/ljspeech."""
    if split == "train":
        selected = entries[:-500]
    elif split == "dev":
        selected = entries[-500:-250]
    elif split == "eval1":
        selected = entries[-250:]
    else:
        known = ", ".join(sorted(_DATASET_CFG))
        raise ValueError(f"Unknown split '{split}'. Expected one of: {known}")

    if not selected:
        raise RuntimeError(f"No utterances selected for split: {split}")

    return selected


class LJSpeechTTSDataset(TorchDataset):
    """LJSpeech TTS dataset that yields speech/text training samples.

    This recipe-local dataset follows the ESPnet3 ``dataset`` module convention.
    The requested ``split`` is sliced directly from ``metadata.csv`` in the raw
    LJSpeech corpus prepared by the ``create_dataset`` stage.

    Args:
        split: Dataset split name. Supported values are ``"train"``,
            ``"dev"``, and ``"eval1"``.
        recipe_dir: Recipe root directory. When omitted, the current recipe
            directory is inferred from this module location.
        include_utt_id: Whether to include the utterance id in each returned
            sample. This is mainly useful during inference.
        corpus_root: Optional existing LJSpeech corpus root. Use this when the
            corpus lives outside the recipe directory.

    Returns:
        A Torch dataset whose items are dicts containing:
        - ``speech``: float32 waveform array
        - ``text``: normalized transcript string
        - ``utt_id``: included only when ``include_utt_id=True``

    Raises:
        ValueError: If ``split`` is unsupported.
        FileNotFoundError: If the resolved raw corpus files do not exist.
        RuntimeError: If the selected split is empty after dataset preparation.

    Notes:
        This dataset assumes ``create_dataset`` has already prepared the raw
        corpus and token list. If you skip that stage, loading fails with
        ``FileNotFoundError`` instead of attempting on-demand dataset repair.

    Examples:
        >>> dataset = LJSpeechTTSDataset(split="train", recipe_dir="egs3/ljspeech/tts")
        >>> sample = dataset[0]
        >>> sorted(sample.keys())
        ['speech', 'text']

        >>> dev_set = LJSpeechTTSDataset(
        ...     split="dev",
        ...     recipe_dir="egs3/ljspeech/tts",
        ...     include_utt_id=True,
        ... )
        >>> "utt_id" in dev_set[0]
        True
    """

    def __init__(
        self,
        split: str,
        recipe_dir: str | Path | None = None,
        include_utt_id: bool = False,
        corpus_root: str | Path | None = None,
    ) -> None:
        self.split = split
        self.include_utt_id = include_utt_id
        recipe_root = (
            Path(recipe_dir).resolve()
            if recipe_dir is not None
            else Path(__file__).resolve().parents[1]
        )

        corpus_root_path = _resolve_corpus_root(recipe_root, corpus_root)
        entries = _read_metadata(corpus_root_path)
        self._entries = _select_split(entries, split)

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        entry = self._entries[int(idx)]
        array, _sr = sf.read(str(entry.wav_path))
        sample = {
            "speech": np.asarray(array, dtype=np.float32),
            "text": entry.text,
        }
        if self.include_utt_id:
            sample["utt_id"] = entry.utt_id
        return sample

"""AmericasNLP 2022 dataset implementation backed by raw corpus directories."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from torch.utils.data import Dataset as TorchDataset

from egs3.americasnlp22.asr.dataset.builder import (
    AmericasNLP22Builder,
    resolve_language_dir,
    resolve_source_root,
)
from espnet3.utils.config_utils import load_config_with_defaults

_CONFIG_RESOURCE = resources.files(__package__).joinpath("config.yaml")
with resources.as_file(_CONFIG_RESOURCE) as _CONFIG_PATH:
    _CONFIG = load_config_with_defaults(str(_CONFIG_PATH), resolve=False)
_DATASET_CFG = _CONFIG["dataset"]

_KNOWN_SPLITS = {str(split) for split in _DATASET_CFG["supported_splits"]}


@dataclass(frozen=True)
class AmericasNLP22Example:
    """Internal index entry derived from one ``meta.tsv`` line.

    ``utt_id`` is kept for deterministic ordering only; samples expose the
    preprocessor-facing fields (``speech``/``text``) exclusively.
    """

    utt_id: str
    audio_path: Path
    text: str


def _scan_split(
    split_dir: Path, max_wav_duration: float | None
) -> list[AmericasNLP22Example]:
    """Build an index for one split by reading ``meta.tsv``.

    The ASR transcript is the ``source_raw`` column (column 3, 1-based),
    exactly as the ESPnet2 recipe used it: punctuation, casing, and
    diacritics are kept verbatim.
    """
    examples: list[AmericasNLP22Example] = []
    meta_path = split_dir / "meta.tsv"
    with meta_path.open("r", encoding="utf-8") as fh:
        header = fh.readline()
        if not header:
            raise RuntimeError(f"Empty meta file: {meta_path}")
        for line_number, raw_line in enumerate(fh, start=2):
            line = raw_line.rstrip("\n")
            if not line:
                continue
            columns = line.split("\t")
            if len(columns) < 3:
                raise RuntimeError(
                    f"Malformed meta.tsv line {line_number} in {meta_path}: "
                    f"expected at least 3 tab-separated columns, got {len(columns)}."
                )
            wav_name = columns[0]
            # Column 3 (`source_raw`) is the transcript; column 2
            # (`source_processed`) is unused, matching the ESPnet2 recipe.
            source_raw = columns[2]
            audio_path = split_dir / wav_name
            if not audio_path.is_file():
                continue
            if max_wav_duration is not None:
                duration = sf.info(str(audio_path)).duration
                if duration > max_wav_duration:
                    continue
            examples.append(
                AmericasNLP22Example(
                    utt_id=Path(wav_name).stem,
                    audio_path=audio_path.resolve(),
                    text=source_raw.strip(),
                )
            )

    if not examples:
        raise RuntimeError(
            f"No usable utterances found under: {split_dir}. "
            "Check that the archive is extracted and meta.tsv is well formed."
        )
    return sorted(examples, key=lambda example: example.utt_id)


class AmericasNLP22Dataset(TorchDataset):
    """Torch dataset reading one language split of the AmericasNLP 2022 corpus.

    Args:
        lang: ISO language code such as ``bzd``. See ``config.yaml`` for the
            supported codes.
        split: ``train`` or ``dev`` (the corpus has no test split; the ESPnet2
            recipe likewise scored the dev set).
        recipe_dir: Recipe root. When omitted, defaults to this module's
            recipe directory.
        source_dir: Optional corpus root override (parent of the per-language
            directories).
        max_wav_duration: Drop utterances longer than this many seconds
            (header-only check). Mirrors the egs2 recipe's
            ``--max_wav_duration 38``.

    Returns:
        ``{"speech": float32 waveform, "text": source_raw transcript}`` — only
        the fields the ESPnet preprocessor accepts; SCP row ids come from the
        item index at inference time.

    Raises:
        ValueError: If ``lang`` or ``split`` is unknown.
        FileNotFoundError: If the resolved corpus root or split directory does
            not exist.
        RuntimeError: If no usable utterances are found for the split.

    Examples:
        >>> dataset = AmericasNLP22Dataset(lang="bzd", split="train")
        >>> sample = dataset[0]
        >>> sorted(sample.keys())
        ['speech', 'text']
    """

    def __init__(
        self,
        lang: str,
        split: str,
        recipe_dir: str | Path | None = None,
        source_dir: str | Path | None = None,
        max_wav_duration: float | None = None,
    ) -> None:
        self.lang = str(lang)
        self.split = str(split)
        if self.split not in _KNOWN_SPLITS:
            known = ", ".join(sorted(_KNOWN_SPLITS))
            raise ValueError(f"Unknown split '{self.split}'. Expected one of: {known}")

        recipe_root = (
            Path(recipe_dir).resolve()
            if recipe_dir is not None
            else Path(__file__).resolve().parents[1]
        )

        builder = AmericasNLP22Builder()
        if not builder.is_source_prepared(
            recipe_dir=recipe_root,
            lang=self.lang,
            source_dir=source_dir,
        ):
            builder.prepare_source(
                recipe_dir=recipe_root,
                lang=self.lang,
                source_dir=source_dir,
            )

        source_root = resolve_source_root(recipe_root, source_dir=source_dir)
        split_dir = resolve_language_dir(source_root, self.lang) / self.split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        self._examples = _scan_split(split_dir, max_wav_duration)

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        example = self._examples[int(idx)]
        array, _sr = sf.read(str(example.audio_path))
        return {
            "speech": np.asarray(array, dtype=np.float32),
            "text": example.text,
        }

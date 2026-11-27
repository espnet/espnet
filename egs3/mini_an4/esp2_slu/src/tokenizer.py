"""Tokenizer inputs for the mini AN4 SLU fixture."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from egs3.mini_an4.esp2_slu.dataset.builder import MiniAn4SluBuilder, resolve_data_root
from egs3.mini_an4.esp2_slu.dataset.dataset import _SPLIT_MANIFEST_PATHS


def _rows(recipe_dir: str | Path | None, splits: Iterable[str]):
    """Yield the manifest rows of ``splits``, building them if needed."""
    recipe_root = (
        Path(recipe_dir).resolve() if recipe_dir is not None else Path.cwd().resolve()
    )
    builder = MiniAn4SluBuilder()
    if not builder.is_built(recipe_dir=recipe_root):
        builder.build(recipe_dir=recipe_root)
    for split in splits:
        manifest = resolve_data_root(recipe_root) / _SPLIT_MANIFEST_PATHS[str(split)]
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                yield line.split("\t", maxsplit=3)


def gather_training_text(
    recipe_dir: str | Path | None = None,
    splits: Iterable[str] = ("train",),
) -> List[str]:
    """Return the transcripts SentencePiece is trained on, without the labels.

    Args:
        recipe_dir: Recipe directory. Defaults to the working directory.
        splits: Manifest splits to read.

    Returns:
        One transcript per utterance.

    Examples:
        >>> gather_training_text(splits=["test"])[:1]  # doctest: +SKIP
        ['go']
    """
    return [transcript for _, _, _, transcript in _rows(recipe_dir, splits)]


def read_intent_labels(
    recipe_dir: str | Path | None = None,
    splits: Iterable[str] = ("train",),
) -> List[str]:
    """Return the intent labels to reserve as SentencePiece symbols.

    Args:
        recipe_dir: Recipe directory. Defaults to the working directory.
        splits: Manifest splits to read.

    Returns:
        The labels, sorted and deduplicated.

    Examples:
        >>> read_intent_labels(splits=["test"])[:1]  # doctest: +SKIP
        ['fash']
    """
    return sorted({intent for _, _, intent, _ in _rows(recipe_dir, splits)})

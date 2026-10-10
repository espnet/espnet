"""Tokenizer text helpers for the AmericasNLP 2022 recipe."""

from __future__ import annotations

from pathlib import Path
from typing import List

from egs3.americasnlp22.asr.dataset.builder import (
    resolve_language_dir,
    resolve_source_root,
)


def gather_training_text(
    lang: str,
    recipe_dir: str | Path | None = None,
    source_dir: str | Path | None = None,
    split: str = "train",
) -> List[str]:
    """Collect transcript text of one language for tokenizer training.

    Mirrors the ESPnet2 recipe's ``--bpe_train_text data/train_<lang>/text``:
    the SentencePiece model is trained on the ``source_raw`` transcripts of
    the single target language, unmodified.

    Args:
        lang: ISO language code, e.g. ``bzd``.
        recipe_dir: Recipe root used to resolve the local download directory.
            When omitted, the recipe directory of this module is used.
        source_dir: Optional corpus root override.
        split: Corpus split to read transcripts from.

    Returns:
        Transcript strings for tokenizer training.

    Raises:
        RuntimeError: If no transcript text is found.
    """
    recipe_root = (
        Path(recipe_dir).resolve()
        if recipe_dir is not None
        else Path(__file__).resolve().parents[1]
    )
    split_path = resolve_language_dir(
        resolve_source_root(recipe_root, source_dir=source_dir), lang
    ) / str(split)
    if not split_path.is_dir():
        raise FileNotFoundError(f"Split not found for tokenizer text: {split_path}")

    texts = []
    meta_path = split_path / "meta.tsv"
    with meta_path.open("r", encoding="utf-8") as fh:
        fh.readline()  # header
        for raw_line in fh:
            columns = raw_line.rstrip("\n").split("\t")
            if len(columns) < 3:
                continue
            text = columns[2].strip()
            if text:
                texts.append(text)
    if not texts:
        raise RuntimeError(f"No transcript text found under: {meta_path}")
    return texts

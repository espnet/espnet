from __future__ import annotations

from pathlib import Path
from typing import List

from egs3.librispeech_100.asr.dataset.builder import resolve_source_root


def _parse_transcript_file(transcript_path: Path) -> list[str]:
    texts: list[str] = []
    with transcript_path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            _, *words = line.split()
            if words:
                texts.append(" ".join(words))
    return texts


def gather_training_text(
    recipe_dir: Path | None = None,
    source_dir: Path | None = None,
    split: str = "train-clean-100",
) -> List[str]:
    """Collect transcript text for tokenizer training.

    Args:
        recipe_dir: Recipe root used to resolve the local download directory.
            When omitted, the current working directory is used.
        source_dir: Optional LibriSpeech parent/root override.
        split: Raw LibriSpeech split name.

    Returns:
        Transcript strings for tokenizer training.

    Raises:
        FileNotFoundError: If the split path cannot be resolved.
        RuntimeError: If no transcript text is found.
    """
    recipe_root = (
        Path(recipe_dir).resolve() if recipe_dir is not None else Path.cwd().resolve()
    )
    split_path = resolve_source_root(recipe_root, source_dir=source_dir) / split
    if not split_path.is_dir():
        raise FileNotFoundError(f"Split not found for tokenizer text: {split_path}")

    texts = []
    for transcript_path in split_path.rglob("*.trans.txt"):
        texts.extend(_parse_transcript_file(transcript_path))
    if not texts:
        raise RuntimeError("No transcript text found for tokenizer training.")
    return texts

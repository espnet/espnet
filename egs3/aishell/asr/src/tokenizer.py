from __future__ import annotations

from pathlib import Path
from typing import List

from egs3.aishell.asr.dataset.builder import resolve_source_root


def gather_training_text(
    recipe_dir: Path | None = None,
    source_dir: Path | None = None,
    split: str = "train",
) -> List[str]:
    """Collect transcript text for tokenizer training.

    Args:
        recipe_dir: Recipe root used to resolve the local download directory.
            When omitted, the current working directory is used.
        source_dir: Optional AISHELL-1 parent/root override.
        split: AISHELL-1 split name. Only ``train`` is appropriate for training.

    Returns:
        Transcript strings for tokenizer training.

    Raises:
        FileNotFoundError: If the split path cannot be resolved.
        RuntimeError: If no transcript text is found.
    """
    recipe_root = (
        Path(recipe_dir).resolve() if recipe_dir is not None else Path.cwd().resolve()
    )
    source_root = resolve_source_root(recipe_root, source_dir=source_dir)
    split_path = source_root / "wav" / split
    if not split_path.is_dir():
        raise FileNotFoundError(f"Split not found for tokenizer text: {split_path}")

    train_ids = {audio_path.stem for audio_path in split_path.rglob("*.wav")}
    transcript_path = source_root / "transcript" / "aishell_transcript_v0.8.txt"
    texts = []
    with transcript_path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            utt_id, separator, text = raw_line.strip().partition(" ")
            if utt_id in train_ids and separator and text:
                texts.append(text.replace(" ", ""))
    if not texts:
        raise RuntimeError("No transcript text found for tokenizer training.")
    return texts

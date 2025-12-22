from __future__ import annotations

import os
from pathlib import Path
from typing import List

from tqdm import tqdm


def _parse_transcript_file(transcript_path: Path) -> list[str]:
    texts: list[str] = []
    with transcript_path.open("r", encoding="utf-8") as f:
        for line in f:
            _, *words = line.strip().split()
            if words:
                texts.append(" ".join(words))
    return texts


def gather_training_text(dataset_dir: Path, split: str = "train-clean-100") -> List[str]:
    """Collect transcript text from a LibriSpeech split for tokenizer training."""
    split_path = Path(dataset_dir) / "LibriSpeech" / split
    if not split_path.exists():
        raise FileNotFoundError(f"Split not found for tokenizer text: {split_path}")

    texts: list[str] = []
    for root, _dirs, files in tqdm(os.walk(split_path)):
        for file in files:
            if file.endswith(".trans.txt"):
                transcript_path = Path(root) / file
                texts.extend(_parse_transcript_file(transcript_path))
    return texts

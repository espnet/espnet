from __future__ import annotations

from pathlib import Path
from typing import List


def gather_training_text(manifest_path: Path) -> List[str]:
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    texts: list[str] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", maxsplit=2)
            if len(parts) != 3:
                raise ValueError(f"Invalid manifest line: {line}")
            texts.append(parts[2])

    if not texts:
        raise RuntimeError(f"No text found in manifest: {manifest_path}")
    return texts

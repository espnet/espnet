"""Locate completed SpeechLM distributed checkpoints."""

from pathlib import Path
from typing import Optional


def latest_checkpoint(output_dir: Path) -> Optional[Path]:
    """Return the highest completed step, ignoring incomplete saves."""
    candidates = []
    for path in (Path(output_dir) / "checkpoints").glob("step_*"):
        step = path.name.removeprefix("step_")
        if step.isdigit() and (path / ".metadata").is_file():
            candidates.append((int(step), path))
    return max(candidates)[1] if candidates else None

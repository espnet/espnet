"""Checkpoint discovery for single-stage TorchTitan runs."""

from pathlib import Path


def latest_checkpoint(exp_dir: Path) -> Path | None:
    """Return the highest numbered step with a committed DCP metadata file."""
    complete = (
        path
        for path in (exp_dir / "checkpoints").glob("step_*")
        if path.name.removeprefix("step_").isdecimal()
        and (path / ".metadata").is_file()
    )
    return max(
        complete, key=lambda path: int(path.name.removeprefix("step_")), default=None
    )

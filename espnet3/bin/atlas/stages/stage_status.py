"""Persist per-trial execution status for recipe stages."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from espnet3.bin.atlas.core.utils import get_utc_now, save_yaml

_STATUS_FILENAME = "stage_status.yaml"


def _read_stage_status(trial_dir: Path) -> dict[str, Any]:
    """Read the stage status file, returning an empty status for old trials."""
    path = trial_dir / _STATUS_FILENAME
    if not path.is_file():
        return {"version": 1, "stages": {}}
    data = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(data, dict) or not isinstance(data.get("stages"), dict):
        raise ValueError(f"Invalid stage status file: {path}")
    return data


def _write_stage_status(trial_dir: Path, data: dict[str, Any]) -> None:
    """Atomically write the per-trial stage status file."""
    data["updated_at"] = get_utc_now()
    save_yaml(trial_dir / _STATUS_FILENAME, data)


def _ensure_stage_statuses(trial_dir: Path, stage_names: list[str]) -> dict[str, Any]:
    """Initialize missing stages as pending and return the updated status."""
    data = _read_stage_status(trial_dir)
    stages = data.setdefault("stages", {})
    changed = False
    for stage_name in stage_names:
        if stage_name not in stages:
            now = get_utc_now()
            stages[stage_name] = {
                "status": "pending",
                "updated_at": now,
                "history": [{"status": "pending", "updated_at": now}],
            }
            changed = True
    if changed:
        _write_stage_status(trial_dir, data)
    return data


def _set_stage_status(
    trial_dir: Path,
    stage_name: str,
    status: str,
    attempt_id: str,
    return_code: int | None = None,
) -> None:
    """Set a stage's current status and append a timestamped history entry."""
    data = _read_stage_status(trial_dir)
    stages = data.setdefault("stages", {})
    entry = stages.setdefault(stage_name, {"history": []})
    now = get_utc_now()
    event: dict[str, Any] = {
        "status": status,
        "attempt_id": attempt_id,
        "updated_at": now,
    }
    if return_code is not None:
        event["return_code"] = return_code
    entry.update(event)
    entry.pop("previous_status", None)
    entry.setdefault("history", []).append(dict(event))
    entry["history"] = entry["history"][-100:]
    _write_stage_status(trial_dir, data)


def _invalidate_stages(trial_dir: Path, stage_names: list[str], reason: str) -> None:
    """Mark affected stages pending while preserving their prior history."""
    data = _read_stage_status(trial_dir)
    stages = data.setdefault("stages", {})
    for stage_name in stage_names:
        now = get_utc_now()
        entry = stages.setdefault(stage_name, {"history": []})
        event = {
            "status": "pending",
            "previous_status": entry.get("status"),
            "reason": reason,
            "updated_at": now,
        }
        entry.update(event)
        entry.setdefault("history", []).append(dict(event))
        entry["history"] = entry["history"][-100:]
    _write_stage_status(trial_dir, data)

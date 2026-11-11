"""Serialization helpers for Atlas."""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import re
import tempfile
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


@contextlib.contextmanager
def file_lock(path: Path):
    """Acquire an advisory file lock, retrying every second while busy.

    Args:
        path: Lock-file path; parent directories are created when needed.

    Returns:
        Iterator[None]: Context manager that releases the lock on exit.

    Example:
        >>> with file_lock(Path("state.lock")):
        ...     pass
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(1)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def to_plain_data(value: Any) -> Any:
    """Convert nested dataclasses/configs/paths to plain JSON-safe values."""
    if is_dataclass(value):
        return {k: to_plain_data(v) for k, v in asdict(value).items()}
    if isinstance(value, DictConfig):
        return OmegaConf.to_container(value, resolve=True)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_plain_data(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain_data(v) for v in value]
    return value


def dumps_json(data: Any, indent: int = 2) -> str:
    """Serialize data to JSON."""
    return json.dumps(to_plain_data(data), ensure_ascii=False, indent=indent)


def loads_json(text: str) -> Any:
    """Parse JSON text."""
    return json.loads(text)


def load_yaml(path: Path) -> Any:
    """Load YAML via OmegaConf and return plain Python data."""
    cfg = OmegaConf.load(path)
    return OmegaConf.to_container(cfg, resolve=True)


def save_yaml(path: Path, data: Any, resolve: bool = True) -> None:
    """Atomically write YAML data."""
    plain = to_plain_data(data)
    cfg = OmegaConf.create(plain)
    yaml_text = OmegaConf.to_yaml(cfg, resolve=resolve)
    atomic_write_text(path, yaml_text)


def atomic_write_text(path: Path, text: str) -> None:
    """Atomically write text."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(dir=path.parent, prefix=".tmp_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(temp_path, path)
    except Exception:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise


def append_text(path: Path, text: str) -> None:
    """Append text with a sibling lock file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(path.with_suffix(path.suffix + ".lock")):
        with path.open("a", encoding="utf-8") as handle:
            handle.write(text)


def flatten_dict(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a nested dict using dotted keys."""
    flat: dict[str, Any] = {}
    for key, value in data.items():
        joined = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_dict(value, joined))
        else:
            flat[joined] = value
    return flat


def unflatten_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Expand dotted keys into nested dicts."""
    nested: dict[str, Any] = {}
    for key, value in data.items():
        current = nested
        parts = str(key).split(".")
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = value
    return nested


def apply_dotted_patch(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """Apply a dotted-key patch onto a nested dict, indexing into lists by position.

    A dotted key part that is a plain non-negative integer (e.g. ``"0"``) is
    treated as a list index whenever the current container is a list, so
    keys like ``dataset.train.0.transform.transforms.1.apply_prob`` can reach
    into a ``dataset.train: [...]``-style list without clobbering it (and its
    sibling entries) into a dict.
    """
    merged = json.loads(json.dumps(base))

    def _set_child(container, part: str, ensure_kind: type):
        if isinstance(container, list):
            idx = int(part)
            while len(container) <= idx:
                container.append({} if ensure_kind is dict else [])
            if not isinstance(container[idx], (dict, list)):
                container[idx] = {} if ensure_kind is dict else []
            return container[idx]
        if part not in container or not isinstance(container[part], (dict, list)):
            container[part] = {} if ensure_kind is dict else []
        return container[part]

    for key, value in patch.items():
        current = merged
        parts = str(key).split(".")
        for part in parts[:-1]:
            current = _set_child(current, part, dict)
        last = parts[-1]
        if isinstance(current, list):
            idx = int(last)
            while len(current) <= idx:
                current.append(None)
            current[idx] = value
        else:
            current[last] = value
    return merged


def pattern_to_regex(pattern: str) -> re.Pattern[str]:
    """Convert a `{value}` placeholder pattern into a regex."""
    escaped = re.escape(pattern).replace(r"\{value\}", r"(?P<value>[-+]?\d*\.?\d+)")
    return re.compile(escaped)

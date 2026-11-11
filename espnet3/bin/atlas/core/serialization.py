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
    """Return the current UTC time as a second-precision ISO-8601 string.

    Args:
        None.

    Returns:
        str: Current UTC time, e.g. ``"2026-01-02T03:04:05+00:00"``.

    Example:
        >>> len(utc_now())
        25
    """
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def to_plain_data(value: Any) -> Any:
    """Recursively convert dataclasses/OmegaConf configs/paths to plain data.

    Used before JSON/YAML serialization so state records, trial objects, and
    resolved configs all become built-in ``dict``/``list``/``str`` types.

    Args:
        value: Any value, possibly nested (dataclass instance, OmegaConf
            ``DictConfig``, ``Path``, ``dict``, ``list``/``tuple``, or a
            plain JSON-safe scalar).

    Returns:
        Any: An equivalent structure built only from JSON-safe types.
        Non-container, non-special values are returned unchanged.

    Example:
        >>> to_plain_data({"p": Path("a/b")})
        {'p': 'a/b'}
    """
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
    """Serialize a value to a JSON string, passing it through ``to_plain_data`` first.

    Args:
        data: Value to serialize; dataclasses/configs/paths are converted to
            plain data before encoding.
        indent: Number of spaces to indent nested structures.

    Returns:
        str: The JSON text.

    Example:
        >>> dumps_json({"a": 1}, indent=0)
        '{\\n"a": 1\\n}'
    """
    return json.dumps(to_plain_data(data), ensure_ascii=False, indent=indent)


def loads_json(text: str) -> Any:
    """Parse a JSON string into Python data.

    Args:
        text: JSON-encoded text.

    Returns:
        Any: The decoded Python value (``dict``, ``list``, or scalar).

    Example:
        >>> loads_json('{"a": 1}')
        {'a': 1}
    """
    return json.loads(text)


def load_yaml(path: Path) -> Any:
    """Load a YAML file through OmegaConf and return resolved plain data.

    Args:
        path: Path to the YAML file.

    Returns:
        Any: The file's content as plain ``dict``/``list``/scalar data, with
        any OmegaConf interpolations resolved.

    Example:
        >>> load_yaml(Path("study/atlas.yaml"))["atlas"]["study_name"]
        'my_study'
    """
    cfg = OmegaConf.load(path)
    return OmegaConf.to_container(cfg, resolve=True)


def save_yaml(path: Path, data: Any, resolve: bool = True) -> None:
    """Atomically write data to a YAML file.

    Converts ``data`` through :func:`to_plain_data` first, so dataclasses,
    OmegaConf configs, and ``Path`` values can be passed directly.

    Args:
        path: Destination YAML file path.
        data: Value to serialize (plain data, dataclass, or OmegaConf config).
        resolve: Whether to resolve OmegaConf interpolations before writing.

    Returns:
        None.

    Example:
        >>> save_yaml(Path("study/atlas.yaml"), {"atlas": {"study_name": "x"}})
    """
    plain = to_plain_data(data)
    cfg = OmegaConf.create(plain)
    yaml_text = OmegaConf.to_yaml(cfg, resolve=resolve)
    atomic_write_text(path, yaml_text)


def atomic_write_text(path: Path, text: str) -> None:
    """Write text to a file atomically via a sibling temp file and rename.

    Writes to a ``.tmp_*`` file in the same directory, then ``os.replace``s
    it onto ``path`` so readers never observe a partially written file. The
    temp file is removed if the write fails.

    Args:
        path: Destination file path; parent directories are created if
            needed.
        text: Text to write (UTF-8 encoded).

    Returns:
        None.

    Example:
        >>> atomic_write_text(Path("study/program.md"), "# Goal\\n")
    """
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
    """Append text to a file, serialized against concurrent writers.

    Holds a :func:`file_lock` on a ``.lock`` sibling of ``path`` for the
    duration of the append, so concurrent node runs (for example several
    workers appending to the same ``trials.csv``) do not interleave writes.

    Args:
        path: File to append to; parent directories are created if needed.
        text: Text to append (UTF-8 encoded).

    Returns:
        None.

    Example:
        >>> append_text(Path("study/state_events.csv"), "row\\n")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(path.with_suffix(path.suffix + ".lock")):
        with path.open("a", encoding="utf-8") as handle:
            handle.write(text)


def flatten_dict(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a nested dict into a single level using dotted keys.

    Args:
        data: Possibly nested mapping to flatten.
        prefix: Dotted key prefix to prepend to every key (used internally
            for recursion; leave as ``""`` for a top-level call).

    Returns:
        dict[str, Any]: A one-level mapping where each key is the dotted
        path to a leaf value in ``data``.

    Example:
        >>> flatten_dict({"a": {"b": 1}})
        {'a.b': 1}
    """
    flat: dict[str, Any] = {}
    for key, value in data.items():
        joined = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_dict(value, joined))
        else:
            flat[joined] = value
    return flat


def unflatten_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Expand a dotted-key mapping back into a nested dict.

    The inverse of :func:`flatten_dict`.

    Args:
        data: Mapping whose keys are dotted paths, e.g. ``{"a.b": 1}``.

    Returns:
        dict[str, Any]: The equivalent nested mapping.

    Example:
        >>> unflatten_dict({"a.b": 1})
        {'a': {'b': 1}}
    """
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

    Args:
        base: Nested dict to patch. Copied (via a JSON round-trip) rather
            than mutated in place.
        patch: Mapping of dotted keys to the values to set at those paths.

    Returns:
        dict[str, Any]: A new dict equal to ``base`` with every ``patch``
        entry applied.

    Example:
        >>> apply_dotted_patch({"trainer": {"lr": 0.1}}, {"trainer.lr": 0.2})
        {'trainer': {'lr': 0.2}}
    """
    merged = json.loads(json.dumps(base))

    def _set_child(container, part: str, ensure_kind: type):
        """Return (creating if absent) the dict/list child at key/index ``part``."""
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
    """Convert a ``{value}`` placeholder pattern into a matching regex.

    Used to parse a numeric metric out of a line of text, e.g. a metric
    extraction config's ``pattern: "WER: {value}"`` matched against a log
    line.

    Args:
        pattern: Literal text containing exactly one ``{value}`` placeholder.

    Returns:
        re.Pattern[str]: A compiled regex with one named group ``value``
        that matches an integer or decimal number where ``{value}`` was.

    Example:
        >>> pattern_to_regex("WER: {value}").search("WER: 12.3").group("value")
        '12.3'
    """
    escaped = re.escape(pattern).replace(r"\{value\}", r"(?P<value>[-+]?\d*\.?\d+)")
    return re.compile(escaped)

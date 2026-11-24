"""Small helpers shared by the Atlas runtime and stages.

Time stamps, atomic file and YAML writes, flattening of metric dicts, and the
helpers that apply and filter the dotted-key config patches agents propose.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def get_utc_now() -> str:
    """Return the current UTC time as a second-precision ISO-8601 string.

    Args:
        None.

    Returns:
        str: Current UTC time, e.g. ``"2026-01-02T03:04:05+00:00"``.

    Example:
        >>> len(get_utc_now())
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


def write_text_atomically(path: Path, text: str) -> None:
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
        >>> write_text_atomically(Path("study/program.md"), "# Goal\\n")
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
    write_text_atomically(path, yaml_text)


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


def apply_dotted_patch(cfg, patch: dict):
    """Set each dotted key of ``patch`` on ``cfg`` in place, and return ``cfg``.

    A key part that is a number indexes into a list, so
    ``dataset.train.0.transform.p`` reaches into a ``dataset.train`` list. A
    value replaces what is there (a dict is not merged), and missing parents
    are created. A list index past the end of its list raises.

    Example:
        >>> cfg = OmegaConf.create({"trainer": {"lr": 0.1}})
        >>> OmegaConf.to_container(apply_dotted_patch(cfg, {"trainer.lr": 0.2}))
        {'trainer': {'lr': 0.2}}
    """
    for key, value in patch.items():
        OmegaConf.update(cfg, key, value, merge=False, force_add=True)
    return cfg


def convert_pairs_to_dict(pairs: list[dict]) -> dict:
    """Turn an agent's ``[{"key": k, "value": v}, ...]`` list into ``{k: v}``.

    Agent replies are schema-constrained, which forbids free-form objects, so
    dotted-key patches and metric maps arrive as lists of key/value pairs. A
    repeated key keeps its last value.

    Example:
        >>> convert_pairs_to_dict([{"key": "optimizer.lr", "value": 0.001}])
        {'optimizer.lr': 0.001}
    """
    return {pair["key"]: pair["value"] for pair in pairs}


def split_patch_by_search_space(patch: dict, search_space) -> tuple[dict, list[str]]:
    """Split a config patch into allowed keys and keys the search space rules out.

    A key is dropped when it matches any ``denied_keys`` pattern, or when
    ``allowed_keys`` is set and it matches none of its patterns. Patterns are
    ``fnmatch`` globs over the dotted key, for example ``optimizer.*``.

    Args:
        patch: Dotted config keys mapped to new values.
        search_space: The ``atlas.search_space`` config node (``allowed_keys`` and
            ``denied_keys`` are both optional).

    Returns:
        tuple[dict, list[str]]: The allowed part of ``patch`` and the removed keys.

    Example:
        >>> space = {"allowed_keys": ["optimizer.*"], "denied_keys": []}
        >>> split_patch_by_search_space({"optimizer.lr": 1, "trainer.x": 2}, space)
        ({'optimizer.lr': 1}, ['trainer.x'])
    """
    allowed = list(getattr(search_space, "allowed_keys", []) or [])
    denied = list(getattr(search_space, "denied_keys", []) or [])
    kept: dict = {}
    removed: list[str] = []
    for key, value in patch.items():
        key = str(key)
        if any(fnmatch(key, p) for p in denied) or (
            allowed and not any(fnmatch(key, p) for p in allowed)
        ):
            removed.append(key)
        else:
            kept[key] = value
    return kept, removed


def read_optional(path: Path) -> str:
    """Read a text file, or return an empty string if it does not exist."""
    return path.read_text(encoding="utf-8") if path.is_file() else ""

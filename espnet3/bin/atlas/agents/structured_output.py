"""Decode complete JSON replies without accepting fragments of broken replies."""

from __future__ import annotations

import json
from typing import Any


def _reject_constant(value: str) -> None:
    """Reject non-JSON numeric constants accepted by Python's default decoder."""
    raise ValueError(f"Non-JSON numeric constant: {value}")


def _read_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate keys instead of silently choosing a configuration value."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _find_outer_objects(text: str) -> list[str]:
    """Find outer objects, retaining incomplete objects and skipping nested ones."""
    objects = []
    start = None
    stack = []
    quoted = False
    escaped = False
    for index, char in enumerate(text):
        if start is None:
            if char in "{[":
                start = index
                stack = [char]
                quoted = escaped = False
            elif char in "}]":
                raise ValueError("Unmatched closing delimiter outside JSON object")
            continue
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "{[":
            stack.append(char)
        elif char in "}]":
            expected = "{" if char == "}" else "["
            if stack[-1] != expected:
                raise ValueError("Mismatched JSON delimiters")
            stack.pop()
            if not stack:
                if text[start] == "{":
                    objects.append(text[start : index + 1])
                start = None
    if start is not None:
        # Never restart decoding inside this incomplete outer container.
        objects.append(text[start:])
    return objects


def decode_response_object(content: str) -> dict[str, Any]:
    """Read an agent's complete JSON object, optionally surrounded by prose/fences.

    Args:
        content: Final agent reply, including any surrounding Markdown or prose.

    Returns:
        The complete object. When examples precede a final object, the final
        outer object is selected. Nested fragments of malformed output are
        never substituted for that object.

    Raises:
        ValueError: The reply is absent, malformed, has duplicate keys or
            nonfinite numbers, or returns a JSON value other than an object.

    Example:
        >>> decode_response_object('Result: {"status": "continue"}')
        {'status': 'continue'}
    """
    text = content.strip().lstrip("\ufeff")
    options = {
        "parse_constant": _reject_constant,
        "object_pairs_hook": _read_object_pairs,
    }
    try:
        value = json.loads(text, **options)
    except json.JSONDecodeError:
        objects = _find_outer_objects(text)
        if not objects:
            raise ValueError("Agent reply contains no JSON object") from None
        try:
            value = json.loads(objects[-1], **options)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid final JSON object: {exc.msg} "
                f"(line {exc.lineno}, column {exc.colno}, offset {exc.pos})"
            ) from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object, got {type(value).__name__}")
    return value

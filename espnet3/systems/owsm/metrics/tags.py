"""Remove OWSM's tags from a text stream."""

from __future__ import annotations

import re
from typing import Iterable, Pattern


def tag_pattern(symbols: Iterable[str]) -> Pattern[str]:
    """Build a pattern matching exactly ``symbols``.

    Longest first, because alternation is ordered and the inventory nests:
    with ``<na>`` tried first, ``<nan>`` loses its ``<na>`` and leaves ``n>``
    behind.
    """
    ordered = sorted({str(symbol) for symbol in symbols}, key=len, reverse=True)
    if not ordered:
        raise ValueError("no symbols to strip")
    return re.compile("|".join(re.escape(symbol) for symbol in ordered))


def strip_tags(text: str, pattern: Pattern[str]) -> str:
    """Drop every tag and collapse the whitespace they leave behind."""
    return " ".join(pattern.sub(" ", text).split())

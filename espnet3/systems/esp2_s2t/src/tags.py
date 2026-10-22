"""Remove OWSM's tags from a text stream."""

from __future__ import annotations

import re
from typing import Any, Iterable, Optional, Pattern

from espnet3.systems.esp2_s2t.src.symbols import load_symbols


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


def configured_pattern(nlsyms: Any, remove_tags: bool) -> Optional[Pattern[str]]:
    """Return the pattern the three rates strip with, or ``None`` to keep tags.

    Args:
        nlsyms: The recipe's symbol inventory -- the same hydra callable,
            file or list that ``tokenizer.nlsyms`` names.
        remove_tags: Whether to strip at all.

    Raises:
        RuntimeError: Tags are to be removed but no inventory was given.

    Examples:
        >>> configured_pattern(["<asr>"], True).pattern
        '<asr>'
        >>> configured_pattern(None, False) is None
        True
    """
    if not remove_tags:
        return None
    if nlsyms is None:
        raise RuntimeError(
            "nlsyms is required to remove tags: pass the recipe's symbol "
            "inventory, the same one tokenizer.nlsyms names. Set "
            "remove_tags: false to score the tags as text instead."
        )
    return tag_pattern(load_symbols(nlsyms))


def strip_tags(text: str, pattern: Pattern[str]) -> str:
    """Drop every tag and collapse the whitespace they leave behind."""
    return " ".join(pattern.sub(" ", text).split())

#!/usr/bin/env python3

# Copyright 2026 ESPnet Developers
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Language-independent normalization steps.

Each step is a factory ``<name>_setup(**kwargs)`` returning a callable
``str -> str``, which is the same shape as VERSA's ``<metric>_setup``. A
step never guesses: everything it does is named in its config, so that the
config echoed into the result is enough to reproduce the score.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Callable, Iterable, Optional

# Unicode categories starting with "P" are punctuation and "S" is symbols.
# Matching by category rather than by an ASCII list is what makes this work
# for Japanese, Chinese and Arabic punctuation as well.
_PUNCT_CATEGORIES = ("P", "S")


def lowercase_setup() -> Callable[[str], str]:
    """Lowercase the text."""
    return lambda text: text.lower()


def uppercase_setup() -> Callable[[str], str]:
    """Uppercase the text.

    sclite's default scoring is case sensitive, and several ESPnet recipes
    upper-case both sides instead of lower-casing them. Reproducing an
    existing number sometimes means matching that choice exactly.
    """
    return lambda text: text.upper()


def remove_punctuation_setup(
    keep: str = "", replace_with: str = ""
) -> Callable[[str], str]:
    """Remove punctuation and symbols.

    Args:
        keep: Characters to keep even though they are punctuation. Apostrophes
            are the usual case: removing them turns "don't" into "dont", which
            changes WER against a reference that kept them.
        replace_with: What to put in place of a removed character. The default
            deletes it, joining the neighbours; " " keeps them apart.
    """
    keep_set = set(keep)

    def _normalize(text: str) -> str:
        return "".join(
            (
                replace_with
                if unicodedata.category(ch).startswith(_PUNCT_CATEGORIES)
                and ch not in keep_set
                else ch
            )
            for ch in text
        )

    return _normalize


def whitespace_setup() -> Callable[[str], str]:
    """Collapse runs of whitespace and strip the ends."""
    return lambda text: re.sub(r"\s+", " ", text).strip()


def remove_tokens_setup(tokens: Optional[Iterable[str]] = None) -> Callable[[str], str]:
    """Delete whole whitespace-separated tokens.

    The usual targets are the non-speech markers a recipe leaves in its
    references: ``<unk>``, ``<noise>``, ``[laughter]``, and so on.
    """
    drop = set(tokens or ["<unk>", "<noise>", "<sos>", "<eos>", "<blank>"])
    return lambda text: " ".join(t for t in text.split() if t not in drop)


def unicode_setup(form: str = "NFKC") -> Callable[[str], str]:
    """Apply a Unicode normal form.

    NFKC folds full-width Latin to half-width, which matters whenever a
    Japanese or Chinese reference and an English-trained model disagree about
    the width of the same character.
    """
    if form not in ("NFC", "NFD", "NFKC", "NFKD"):
        raise ValueError(f"unknown Unicode normal form: {form}")
    return lambda text: unicodedata.normalize(form, text)

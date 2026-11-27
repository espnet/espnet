#!/usr/bin/env python3

# Copyright 2026 ESPnet Developers
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Language-independent normalization steps.

Each step is a factory ``<name>_setup(**kwargs)`` returning a callable
``str -> str``, which is the same shape as VERSA's ``<metric>_setup``. A
step never guesses: everything it does is named in its config, so that the
config echoed into the result is enough to reproduce the result.
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

    sclite's default comparison is case sensitive, and several ESPnet recipes
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


#: What a recipe means by nlsyms when it does not say.
DEFAULT_NON_SPEECH = ("<unk>", "<noise>", "<sos>", "<eos>", "<blank>")

MATCH_CHOICES = ("token", "substring")


def remove_tokens_setup(
    tokens: Optional[Iterable[str]] = None,
    match: str = "token",
    tokens_file: Optional[str] = None,
) -> Callable[[str], str]:
    """Delete non-speech markers such as ``<unk>``, ``<noise>``, ``[laughter]``.

    ESPnet removes these inside its tokenizers, and the two tokenizers do it
    differently. Which one a recipe used decides its denominator, so the
    choice is explicit here rather than guessed from the metric.

    Args:
        tokens: The markers to delete. None means
            :data:`DEFAULT_NON_SPEECH` unless ``tokens_file`` is given; an
            empty list means delete nothing.
        match: ``token`` matches a whole whitespace-separated token, which is
            what ``espnet2.text.word_tokenizer`` does and what WER wants.
            ``substring`` matches anywhere in the text, which is what
            ``espnet2.text.char_tokenizer`` does and what CER wants. Its
            space handling is copied from there too: removing a marker from
            between two spaces would leave both behind and the CER
            denominator would count them, so when anything is removed, runs
            of spaces collapse and leading and trailing ones go. When
            nothing is removed the text is left alone, doubled spaces
            included -- CharTokenizer._merge_spaces runs under `if removed`,
            and a normalizer that tidied unconditionally would report a
            different denominator than the recipe it is reproducing.
        tokens_file: An ``nlsyms_txt`` file, one marker per line, as 33 egs2
            recipes point at. Read the way ESPnet reads it: trailing
            whitespace stripped, blank lines dropped. Combined with ``tokens``
            when both are given.

    Returns:
        The normalizer.

    Raises:
        ValueError: If ``match`` is not one of :data:`MATCH_CHOICES`.
    """
    if match not in MATCH_CHOICES:
        raise ValueError(f"unknown match '{match}'; expected {list(MATCH_CHOICES)}")

    drop = set()
    if tokens_file is not None:
        with open(tokens_file, encoding="utf-8") as handle:
            drop.update(line.rstrip() for line in handle)
    if tokens is not None:
        drop.update(tokens)
    elif tokens_file is None:
        drop.update(DEFAULT_NON_SPEECH)
    # An empty marker matches at every position, so a scan over it would never
    # advance. ESPnet discards it for the same reason.
    drop.discard("")

    if match == "token":
        return lambda text: " ".join(t for t in text.split() if t not in drop)

    # Longest first: a shorter marker can be a prefix of a longer one, and
    # matching the short one would leave the rest of the long one behind.
    ordered = tuple(sorted(drop, key=len, reverse=True))

    def _remove(text: str) -> str:
        if not ordered:
            return text
        out = []
        removed = False
        index = 0
        while index < len(text):
            for marker in ordered:
                if text.startswith(marker, index):
                    index += len(marker)
                    removed = True
                    break
            else:
                out.append(text[index])
                index += 1
        if not removed:
            return text
        return re.sub(" +", " ", "".join(out)).strip(" ")

    return _remove


def unicode_setup(form: str = "NFKC") -> Callable[[str], str]:
    """Apply a Unicode normal form.

    NFKC folds full-width Latin to half-width, which matters whenever a
    Japanese or Chinese reference and an English-trained model disagree about
    the width of the same character.
    """
    if form not in ("NFC", "NFD", "NFKC", "NFKD"):
        raise ValueError(f"unknown Unicode normal form: {form}")
    return lambda text: unicodedata.normalize(form, text)

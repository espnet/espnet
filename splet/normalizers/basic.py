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

    Note that case folding is usually not a normalization step here: sclite
    folds case itself unless given ``-s``, so the error-rate metric does it
    at scoring time and records which it did (``case="fold"``, the default).
    This step is for pipelines that need the text itself upper-cased, such as
    a corpus whose reference is upper-case and whose score is reported
    case-sensitively.
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


_DEFAULT_REMOVED_TOKENS = ["<unk>", "<noise>", "<sos>", "<eos>", "<blank>"]

MATCH_CHOICES = ("token", "substring")


def _read_symbol_file(path: str) -> list:
    """Read an ESPnet ``nlsyms_txt`` file: one symbol per line.

    espnet2 reads this as ``set(line.rstrip() for line in f)``, which puts an
    empty string in the set for a blank line. An empty symbol matches at every
    position, so blank lines are skipped here instead.
    """
    with open(path, encoding="utf-8") as handle:
        return [line.rstrip("\n").rstrip() for line in handle if line.strip()]


def remove_tokens_setup(
    tokens: Optional[Iterable[str]] = None,
    tokens_file: Optional[str] = None,
    match: str = "token",
) -> Callable[[str], str]:
    """Delete non-linguistic symbols.

    This is what ``asr.sh`` does by passing ``--non_linguistic_symbols
    ${nlsyms_txt} --remove_non_linguistic_symbols true`` to
    ``espnet2.bin.tokenize_text``, and 33 egs2 recipes set an ``nlsyms_txt``.
    espnet2 does the removal inside the tokenizer, and the word and character
    tokenizers do it differently, so ``match`` picks which one to reproduce.

    Args:
        tokens: Symbols to remove. Defaults to the common markers when no
            symbols and no file are given.
        tokens_file: An ESPnet ``nlsyms_txt`` file, one symbol per line.
            Combined with ``tokens`` when both are given.
        match: ``"token"`` drops a whitespace-delimited token when the whole
            token is a symbol, which is what ``WordTokenizer`` does
            (``espnet2/text/word_tokenizer.py:44-51``), and is the right mode
            for WER. ``"substring"`` removes a symbol wherever it occurs,
            which is what ``CharTokenizer`` does
            (``espnet2/text/char_tokenizer.py:48-66``), and is the right mode
            for CER, where a symbol sits inside the character stream.

    Returns:
        A callable removing the symbols from a string.

    Raises:
        ValueError: If ``match`` is not one of the two accepted values.
        FileNotFoundError: If ``tokens_file`` does not exist. espnet2 only
            warns and carries on with an empty set, which silently scores
            something other than what was asked for.

    Note:
        ``CharTokenizer`` iterates a *set* of symbols and takes the first that
        matches, so where one symbol is a prefix of another its choice depends
        on set iteration order. This uses longest match first, which is
        deterministic and agrees with it whenever no symbol is a prefix of
        another -- true of every nlsyms list in egs2, including the
        1702-symbol OWSM one, where every symbol is a complete ``<...>``.
    """
    if match not in MATCH_CHOICES:
        raise ValueError(f"unknown match '{match}'. Available: {sorted(MATCH_CHOICES)}")

    symbols = list(tokens) if tokens else []
    if tokens_file is not None:
        symbols += _read_symbol_file(tokens_file)
    if not symbols:
        symbols = list(_DEFAULT_REMOVED_TOKENS)
    drop = {symbol for symbol in symbols if symbol}

    if match == "token":
        return lambda text: " ".join(t for t in text.split() if t not in drop)

    # Longest first, so the alternation prefers the longer symbol where one
    # is a prefix of another. Whitespace is deliberately left alone: removing
    # "<noise>" from "a <noise> b" has to leave two spaces, because that is
    # what CharTokenizer produces and therefore what the CER denominator
    # counts.
    pattern = re.compile(
        "|".join(re.escape(s) for s in sorted(drop, key=len, reverse=True))
    )
    return lambda text: pattern.sub("", text)


def unicode_setup(form: str = "NFKC") -> Callable[[str], str]:
    """Apply a Unicode normal form.

    NFKC folds full-width Latin to half-width, which matters whenever a
    Japanese or Chinese reference and an English-trained model disagree about
    the width of the same character.
    """
    if form not in ("NFC", "NFD", "NFKC", "NFKD"):
        raise ValueError(f"unknown Unicode normal form: {form}")
    return lambda text: unicodedata.normalize(form, text)

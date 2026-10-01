#!/usr/bin/env python3

# Copyright 2026 ESPnet Developers
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Word and character error rate.

One implementation over two unit types, because WER and CER are the same
computation on different tokens. Each result carries its counts next to its
rate so that the corpus figure is pooled rather than averaged; see
``splet/metrics.py`` for why that is not cosmetic.

The defaults are the ones that reproduce an egs2 recipe: sclite's cost model
and sclite's case folding. ``sclite`` compares case-insensitively unless it
is given ``-s``, which only four corpora in egs2 pass, so a case-sensitive
default would disagree with almost every published ESPnet number.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from splet.alignment import levenshtein_alignment
from splet.normalizers import build_normalizer


def _word_tokenizer() -> Callable[[str], List[str]]:
    """Split on whitespace."""
    return lambda text: text.split()


def _char_tokenizer(
    remove_space: bool = False, space_symbol: Optional[str] = None
) -> Callable[[str], List[str]]:
    """Split into characters.

    Args:
        remove_space: Drop spaces instead of counting them as characters.
            The default keeps them, which is what ``jiwer.cer`` does and
            therefore what the current espnet3 CER reports. Recipes that
            score CER with spaces removed need this set to True to reproduce
            their published numbers.
        space_symbol: Emit this token for a space instead of the space
            character itself. ``"<space>"`` is what ``asr.sh`` scores CER
            with, via espnet2's CharTokenizer. It is one token either way, so
            this changes what an alignment looks like, not what it counts.
    """

    def _tokenize(text: str) -> List[str]:
        if remove_space:
            return list("".join(text.split()))
        if space_symbol is None:
            return list(text)
        return [space_symbol if char == " " else char for char in text]

    return _tokenize


def _noascii_tokenizer() -> Callable[[str], List[str]]:
    """Split non-ASCII words into characters and keep ASCII words whole.

    This is sclite's ``-c NOASCII``, which egs2/seame scores code-switched
    Mandarin/English with: a Chinese run is counted per character, because
    whitespace there is not a word boundary, while an English word stays one
    token.
    """

    def _tokenize(text: str) -> List[str]:
        tokens: List[str] = []
        for word in text.split():
            if word.isascii():
                tokens.append(word)
            else:
                tokens.extend(word)
        return tokens

    return _tokenize


TOKENIZER_CHOICES: Dict[str, Callable[..., Callable[[str], List[str]]]] = {
    "word": _word_tokenizer,
    "char": _char_tokenizer,
    "noascii": _noascii_tokenizer,
}


CASE_CHOICES = ("fold", "sensitive")


def error_rate_setup(
    name: str = "wer",
    tokenizer: str = "word",
    tokenizer_conf: Optional[Dict[str, Any]] = None,
    normalize: Optional[list] = None,
    backend: str = "python",
    costs: str = "sclite",
    case: str = "fold",
    optional_deletion_is_correct: bool = False,
    keep_alignment: bool = False,
) -> Dict[str, Any]:
    """Prepare an error-rate scorer.

    Args:
        name: Prefix for the reported keys. ``wer`` reports ``wer``,
            ``wer_errors``, ``wer_ref_len`` and the S/D/I/C counts.
        tokenizer: Unit to count: ``word`` for WER, ``char`` for CER.
        tokenizer_conf: Keyword arguments for the tokenizer.
        normalize: Normalization pipeline config, applied to both sides.
        backend: Alignment backend. The default is the reference
            implementation; see
            :func:`splet.alignment.levenshtein_alignment`.
        costs: Alignment cost model, ``"sclite"`` (the default) or
            ``"unit"``. This decides the S/D/I split, and on noisy output it
            can also move the total; see :mod:`splet.alignment`.
        case: ``"fold"`` (the default) compares case-insensitively, as sclite
            does. ``"sensitive"`` is sclite's ``-s``.
        optional_deletion_is_correct: Score a deleted ``(word)`` as correct.
            This is sclite's ``-D``.
        keep_alignment: Include the rendered alignment in every result.
            Useful for a handful of utterances, ruinous for a corpus of
            them, so it is off by default.

    Returns:
        The scorer state passed back into :func:`error_rate_metric`.

    Raises:
        ValueError: If the tokenizer or case name is unknown.
    """
    if tokenizer not in TOKENIZER_CHOICES:
        raise ValueError(
            f"unknown tokenizer '{tokenizer}'. Available: {sorted(TOKENIZER_CHOICES)}"
        )
    if case not in CASE_CHOICES:
        raise ValueError(f"unknown case '{case}'. Available: {sorted(CASE_CHOICES)}")
    return {
        "name": name,
        "tokenizer": TOKENIZER_CHOICES[tokenizer](**(tokenizer_conf or {})),
        "normalizer": build_normalizer(normalize),
        "backend": backend,
        "costs": costs,
        "case": case,
        "optional_deletion_is_correct": optional_deletion_is_correct,
        "keep_alignment": keep_alignment,
    }


def error_rate_metric(
    scorer: Dict[str, Any],
    pred_text: str,
    gt_text: str,
) -> Dict[str, Any]:
    """Score one hypothesis against one reference.

    Args:
        scorer: State from :func:`error_rate_setup`.
        pred_text: Hypothesis text.
        gt_text: Reference text.

    Returns:
        The rate, the counts it was computed from, and optionally the
        rendered alignment. Every key is prefixed with the scorer's name.
    """
    name: str = scorer["name"]
    normalizer = scorer["normalizer"]
    tokenize = scorer["tokenizer"]

    def prepare(text: str) -> List[str]:
        text = normalizer(text)
        # After normalization, so that a configured pipeline sees the text as
        # written; sclite folds case at alignment time, not before cleaning.
        if scorer["case"] == "fold":
            text = text.lower()
        return tokenize(text)

    alignment = levenshtein_alignment(
        prepare(gt_text),
        prepare(pred_text),
        backend=scorer["backend"],
        costs=scorer["costs"],
        optional_deletion_is_correct=scorer["optional_deletion_is_correct"],
    )

    result: Dict[str, Any] = {
        name: alignment.error_rate,
        f"{name}_errors": alignment.errors,
        f"{name}_ref_len": alignment.ref_len,
        f"{name}_hyp_len": alignment.hyp_len,
        f"{name}_sub": alignment.substitutions,
        f"{name}_del": alignment.deletions,
        f"{name}_ins": alignment.insertions,
        f"{name}_hit": alignment.hits,
    }
    if scorer["keep_alignment"]:
        result["alignment"] = alignment.to_string()
    return result

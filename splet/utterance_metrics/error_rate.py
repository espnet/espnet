#!/usr/bin/env python3

# Copyright 2026 ESPnet Developers
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Word and character error rate.

One implementation over two unit types, because WER and CER are the same
computation on different tokens. Each result carries its counts next to its
rate so that the corpus figure is pooled rather than averaged; see
``splet/metrics.py`` for why that is not cosmetic.

This is the one metric the skeleton implements, to pin down the contract
every other metric follows. Its S/D/I split has not yet been checked against
sclite -- see :mod:`splet.alignment` and espnet/espnet#6760.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from splet.alignment import levenshtein_alignment
from splet.normalizers import build_normalizer


def _word_tokenizer() -> Callable[[str], List[str]]:
    """Split on whitespace."""
    return lambda text: text.split()


def _char_tokenizer(remove_space: bool = False) -> Callable[[str], List[str]]:
    """Split into characters.

    Args:
        remove_space: Drop spaces instead of counting them as characters.
            The default keeps them, which is what ``jiwer.cer`` does and
            therefore what the current espnet3 CER reports. Recipes that
            score CER with spaces removed need this set to True to reproduce
            their published numbers.
    """

    def _tokenize(text: str) -> List[str]:
        return list("".join(text.split()) if remove_space else text)

    return _tokenize


TOKENIZER_CHOICES: Dict[str, Callable[..., Callable[[str], List[str]]]] = {
    "word": _word_tokenizer,
    "char": _char_tokenizer,
}


def error_rate_setup(
    name: str = "wer",
    tokenizer: str = "word",
    tokenizer_conf: Optional[Dict[str, Any]] = None,
    normalize: Optional[list] = None,
    backend: str = "python",
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
            implementation, whose S/D/I split does not depend on which
            packages happen to be installed; see
            :func:`splet.alignment.levenshtein_alignment`.
        keep_alignment: Include the rendered alignment in every result.
            Useful for a handful of utterances, ruinous for a corpus of
            them, so it is off by default.

    Returns:
        The scorer state passed back into :func:`error_rate_metric`.

    Raises:
        ValueError: If the tokenizer name is unknown.
    """
    if tokenizer not in TOKENIZER_CHOICES:
        raise ValueError(
            f"unknown tokenizer '{tokenizer}'. Available: {sorted(TOKENIZER_CHOICES)}"
        )
    return {
        "name": name,
        "tokenizer": TOKENIZER_CHOICES[tokenizer](**(tokenizer_conf or {})),
        "normalizer": build_normalizer(normalize),
        "backend": backend,
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

    alignment = levenshtein_alignment(
        tokenize(normalizer(gt_text)),
        tokenize(normalizer(pred_text)),
        backend=scorer["backend"],
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

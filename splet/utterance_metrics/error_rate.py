#!/usr/bin/env python3

# Copyright 2026 ESPnet Developers
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Word and character error rate.

One implementation over two unit types, because WER and CER are the same
computation on different tokens. Each result carries its counts next to its
rate so that the corpus figure is pooled rather than averaged; see
``splet/summary.py`` for why that is not cosmetic.

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
            measure CER with spaces removed need this set to True to reproduce
            their published numbers.
    """

    def _tokenize(text: str) -> List[str]:
        return list("".join(text.split()) if remove_space else text)

    return _tokenize


TOKENIZER_CHOICES: Dict[str, Callable[..., Callable[[str], List[str]]]] = {
    "word": _word_tokenizer,
    "char": _char_tokenizer,
}

#: The result keys an error rate reports, by the suffix after its configured
#: id, and how each is reduced over the corpus (see ``splet/summary.py``):
#: the rate is recomputed from the summed error and reference-length counts,
#: the counts are summed, the rendered alignment is text.
OUTPUTS: Dict[str, str] = {
    "": "pool:_errors/_ref_len",
    "_errors": "sum",
    "_ref_len": "sum",
    "_hyp_len": "sum",
    "_sub": "sum",
    "_del": "sum",
    "_ins": "sum",
    "_hit": "sum",
    "_alignment": "text",
}


def error_rate_setup(
    metric_id: str = "wer",
    tokenizer: str = "word",
    tokenizer_conf: Optional[Dict[str, Any]] = None,
    normalize: Optional[list] = None,
    backend: str = "python",
    keep_alignment: bool = False,
) -> Dict[str, Any]:
    """Prepare an error-rate state.

    Args:
        metric_id: The configured id, and so the prefix of every reported
            key: ``wer`` reports ``wer``, ``wer_errors``, ``wer_ref_len``,
            the S/D/I/C counts and ``wer_alignment``. The registry passes the
            config entry's ``id`` (its ``name`` when no id is given), so two
            configurations of this one implementation - raw and normalized,
            say - report under two ids and never overwrite each other.
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
        The state state passed back into :func:`error_rate_metric`.

    Raises:
        ValueError: If the tokenizer name is unknown.
    """
    if tokenizer not in TOKENIZER_CHOICES:
        raise ValueError(
            f"unknown tokenizer '{tokenizer}'. Available: {sorted(TOKENIZER_CHOICES)}"
        )
    return {
        "metric_id": metric_id,
        "tokenizer": TOKENIZER_CHOICES[tokenizer](**(tokenizer_conf or {})),
        "normalizer": build_normalizer(normalize),
        "backend": backend,
        "keep_alignment": keep_alignment,
    }


def error_rate_metric(
    state: Dict[str, Any],
    pred_text: str,
    gt_text: str,
) -> Dict[str, Any]:
    """Measure one hypothesis against one reference.

    Args:
        state: State from :func:`error_rate_setup`.
        pred_text: Hypothesis text.
        gt_text: Reference text.

    Returns:
        The rate, the counts it was computed from, and optionally the
        rendered alignment. Every key is prefixed with the configured id,
        as :data:`OUTPUTS` declares.
    """
    name: str = state["metric_id"]
    normalizer = state["normalizer"]
    tokenize = state["tokenizer"]

    alignment = levenshtein_alignment(
        tokenize(normalizer(gt_text)),
        tokenize(normalizer(pred_text)),
        backend=state["backend"],
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
    if state["keep_alignment"]:
        result[f"{name}_alignment"] = alignment.to_string()
    return result

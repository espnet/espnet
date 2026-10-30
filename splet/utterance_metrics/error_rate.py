#!/usr/bin/env python3

# Copyright 2026 ESPnet Developers
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Word and character error rate.

One implementation over two unit types, because WER and CER are the same
computation on different tokens. Each result carries its counts next to its
rate so that the corpus figure is pooled rather than averaged; see
``splet/summary.py`` for why that is not cosmetic.

This is the one metric the skeleton implements, to pin down the contract
every other metric follows. It aligns with sclite's cost model by default,
and reproduces sclite's counts; :mod:`splet.alignment` lists the two
behaviours of sclite's that it does not have.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from splet.alignment import levenshtein_alignment, resolve_backend, resolve_costs
from splet.normalizers import build_normalizer

try:
    import sentencepiece as _sentencepiece
except ImportError:  # pragma: no cover - exercised by the no-extras CI job
    _sentencepiece = None


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
            measure CER with spaces removed need this set to True to reproduce
            their published numbers.
        space_symbol: Emit this token for a space instead of the space
            itself. ``espnet2.text.char_tokenizer`` uses ``<space>``, and
            a recipe scored through ``tokenize_text`` counts that symbol
            in its CER denominator.

    Raises:
        ValueError: If both arguments are given, which asks for a space
            to be dropped and kept at once.
    """
    if remove_space and space_symbol is not None:
        raise ValueError(
            "remove_space and space_symbol cannot both be set: one drops "
            "the space, the other keeps it under a name"
        )

    def _tokenize(text: str) -> List[str]:
        if remove_space:
            return list("".join(text.split()))
        if space_symbol is None:
            return list(text)
        return [space_symbol if ch == " " else ch for ch in text]

    return _tokenize


def _noascii_tokenizer() -> Callable[[str], List[str]]:
    """Split into characters, but keep runs of ASCII whole.

    sclite's ``-c NOASCII``, whose documentation says it "does not split
    ASCII words". It is finer than that, which matters for code-switched
    text: the split is per run, not per word, so ``AI的`` is two tokens
    (``AI``, ``的``) rather than one or three. Verified against sclite
    2.10, which also keeps trailing punctuation in the ASCII run.
    """

    def _tokenize(text: str) -> List[str]:
        tokens: List[str] = []
        for word in text.split():
            run = ""
            for character in word:
                if character.isascii():
                    run += character
                    continue
                if run:
                    tokens.append(run)
                    run = ""
                tokens.append(character)
            if run:
                tokens.append(run)
        return tokens

    return _tokenize


def _sentencepiece_tokenizer(model: str) -> Callable[[str], List[str]]:
    """Split into the pieces of a SentencePiece model.

    Args:
        model: Path to the ``.model`` file. A token error rate is only
            comparable against another one from the same model, so there
            is no default.

    Raises:
        ImportError: If sentencepiece is not installed.
    """
    if _sentencepiece is None:
        raise ImportError(
            "tokenizer='sentencepiece' requires sentencepiece: "
            "pip install sentencepiece"
        )
    processor = _sentencepiece.SentencePieceProcessor()
    processor.Load(model)
    return lambda text: processor.EncodeAsPieces(text)


TOKENIZER_CHOICES: Dict[str, Callable[..., Callable[[str], List[str]]]] = {
    "word": _word_tokenizer,
    "char": _char_tokenizer,
    "noascii": _noascii_tokenizer,
    "sentencepiece": _sentencepiece_tokenizer,
}

#: How a configured case policy folds the text it is given. ``fold`` is
#: sclite's default; ``sensitive`` is its ``-s``.
CASE_CHOICES: Dict[str, Callable[[str], str]] = {
    "fold": str.casefold,
    "sensitive": lambda text: text,
}

#: The result keys an error rate reports, by the suffix after its configured
#: id, and how each is reduced over the corpus (see ``splet/summary.py``):
#: the rate is recomputed from the summed error and reference-length counts,
#: the counts are summed, the rendered alignment is text.
OUTPUTS: Dict[str, str] = {
    "": "rate",
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
    costs: str = "sclite",
    case: str = "fold",
    optional_deletion_is_correct: bool = False,
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
        costs: Alignment cost model. ``sclite`` (the default) reproduces
            sclite's counts; ``unit`` is plain edit distance, which is what
            jiwer reports. See :mod:`splet.alignment`.
        case: ``fold`` (the default) case-folds both sides before
            tokenizing, which is what sclite does unless given ``-s``;
            ``sensitive`` is ``-s``. Folded text is what a rendered
            alignment shows, because it is what was scored.
        optional_deletion_is_correct: sclite's ``-D``. Score a token
            written ``(word)`` as correct when it is dropped rather than
            as an error. A reference that does not use the marking is
            unaffected either way.
        backend: Alignment backend. The default is the reference
            implementation, whose S/D/I split does not depend on which
            packages happen to be installed; see
            :func:`splet.alignment.levenshtein_alignment`. ``auto`` is
            resolved here, so the state, and the metadata copied from it,
            names the backend that actually runs.
        keep_alignment: Include the rendered alignment in every result.
            Useful for a handful of utterances, ruinous for a corpus of
            them, so it is off by default.

    Returns:
        The state state passed back into :func:`error_rate_metric`.

    Raises:
        ValueError: If the tokenizer, cost model, case policy or backend
            name is unknown.
        ImportError: If ``backend="rapidfuzz"`` and it is not installed.
    """
    if tokenizer not in TOKENIZER_CHOICES:
        raise ValueError(
            f"unknown tokenizer '{tokenizer}'. Available: {sorted(TOKENIZER_CHOICES)}"
        )
    if case not in CASE_CHOICES:
        raise ValueError(
            f"unknown case policy '{case}'. Available: {sorted(CASE_CHOICES)}"
        )
    return {
        "metric_id": metric_id,
        "tokenizer": TOKENIZER_CHOICES[tokenizer](**(tokenizer_conf or {})),
        "normalizer": build_normalizer(normalize),
        "costs": costs,
        "cost_model": resolve_costs(costs),
        "case": case,
        "fold": CASE_CHOICES[case],
        "optional_deletion_is_correct": optional_deletion_is_correct,
        "backend": resolve_backend(backend, costs),
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
    fold = state["fold"]

    alignment = levenshtein_alignment(
        tokenize(fold(normalizer(gt_text))),
        tokenize(fold(normalizer(pred_text))),
        backend=state["backend"],
        costs=state["cost_model"],
        optional_deletion_is_correct=state["optional_deletion_is_correct"],
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

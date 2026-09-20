"""Shared IPA helpers for the phone recognition metrics.

Both :class:`espnet3.systems.pr.metrics.per.PER` and
:class:`espnet3.systems.pr.metrics.pfer.PFER` score concatenated IPA strings
rather than whitespace-delimited tokens, so they need the same two steps:
normalize a transcript, then split it into phones. Keeping those here means the
two metrics cannot drift apart, which would silently change every score.
"""

from __future__ import annotations

import string
import unicodedata
from typing import List

try:
    import panphon.distance
except ImportError:
    panphon = None

_PUNCTUATION = str.maketrans("", "", string.punctuation)

# panphon builds a feature table from a bundled CSV on construction, which is
# slow enough to matter once per utterance. One instance is enough: Distance is
# read-only once built.
_DISTANCE = None


def require_panphon() -> None:
    """Raise if the optional panphon dependency is missing.

    Raises:
        RuntimeError: If ``panphon`` is not installed, with the install command.
    """
    if panphon is None:
        raise RuntimeError(
            "panphon is required to score phone recognition. "
            "Please install it with `pip install espnet[pr]`."
        )


def get_distance() -> "panphon.distance.Distance":
    """Return the shared panphon ``Distance``, building it on first use.

    Returns:
        A process-wide ``panphon.distance.Distance`` instance.

    Raises:
        RuntimeError: If ``panphon`` is not installed.
    """
    global _DISTANCE
    require_panphon()
    if _DISTANCE is None:
        _DISTANCE = panphon.distance.Distance()
    return _DISTANCE


def clean_ipa_text(text: str) -> str:
    """Normalize an IPA transcript so hypotheses and references are comparable.

    The steps, in this order, are what the PhoneticXeus evaluator applies:
    drop spaces, drop ASCII punctuation, decompose to NFD, and rewrite Latin
    ``g`` (U+0067) as IPA ``ɡ`` (U+0261). Spaces go because word boundaries are
    not scored; punctuation goes because it removes the ``/`` separators a model
    may emit. Note that this also strips ``<`` and ``>``, so special tokens must
    already have been removed -- otherwise ``<unk>`` survives as the letters
    ``unk`` and is scored as three phones.

    Args:
        text: Raw IPA transcript.

    Returns:
        The normalized transcript, possibly empty.

    Examples:
        >>> clean_ipa_text("ð/ɪ/s")
        'ðɪs'
        >>> clean_ipa_text("go")
        'ɡo'
    """
    stripped = text.replace(" ", "").translate(_PUNCTUATION)
    return unicodedata.normalize("NFD", stripped).replace("g", "ɡ").strip()


def segment_ipa(text: str) -> List[str]:
    """Split a normalized IPA string into phones.

    Uses panphon's greedy longest-match segmenter, which keeps a base glyph
    together with its diacritics, so ``aː`` and ``t͡ʃ`` are one phone each.
    Characters panphon does not recognize are dropped. This is deliberately not
    a whitespace or ``/`` split: both sides are re-segmented the same way, so a
    hypothesis and a reference are compared on identical units no matter how
    either was tokenized upstream.

    Args:
        text: IPA string, normally the output of :func:`clean_ipa_text`.

    Returns:
        The phones, in order. Empty when nothing is recognizable.

    Raises:
        RuntimeError: If ``panphon`` is not installed.

    Examples:
        >>> segment_ipa("t͡ʃaː")
        ['t͡ʃ', 'aː']
    """
    return get_distance().fm.ipa_segs(text)

#!/usr/bin/env python3

# Copyright 2026 ESPnet Developers
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Whisper-style normalization, as egs2 applies it at scoring time.

``asr.sh`` cleans its text with ``espnet2.text.cleaner.TextCleaner``, whose
``whisper_en`` and ``whisper_basic`` types are openai-whisper's
``EnglishTextNormalizer`` and ``BasicTextNormalizer``. Those two are the only
cleaners any egs2 ASR-family recipe sets -- aishell's Whisper fine-tune,
allsstar_eng/s2t1, and the documented eval setting for owsm_v2 and owsm_v3 --
so they are the ones that can change a published number.

The implementation is vendored, not reimplemented and not imported:

* Not imported, because ``from whisper.normalizers import ...`` executes
  ``whisper/__init__.py``, which imports torch. ``ci/check_splet_independence.py``
  forbids that, and rightly: a text evaluator that drags in a tensor library
  cannot be extracted into its own package.
* Not reimplemented, because the normalizer is four passes -- a 1739-entry
  spelling table, a number formatter, a contraction expander and a
  currency/unit handler -- each worth a point or more of WER on its own, with
  behaviour nobody would guess. It rewrites ``Mr. O'Brien`` to
  ``mister 0 brien``: the number pass reads a standalone "o" as a zero. A
  reimplementation that is merely reasonable would produce a number
  comparable with nothing.

``test/splet/test_whisper_normalizer.py`` asserts string equality against the
installed openai-whisper, so the copy is checked rather than trusted.
"""

from __future__ import annotations

from typing import Callable

from ._basic import BasicTextNormalizer
from ._english import EnglishTextNormalizer

__all__ = ["BasicTextNormalizer", "EnglishTextNormalizer", "whisper_setup"]

# TextCleaner's names for these, so a SPLET config can say what asr.sh says.
_FLAVOURS = {"en", "english", "whisper_en", "basic", "whisper_basic"}


def whisper_setup(language: str = "en", **kwargs) -> Callable[[str], str]:
    """Build a Whisper text normalizer.

    Args:
        language: ``"en"`` (the default) selects ``EnglishTextNormalizer``,
            which is TextCleaner's ``whisper_en``. ``"basic"`` selects
            ``BasicTextNormalizer``, TextCleaner's ``whisper_basic``, which is
            what the non-English recipes use. The spellings ``whisper_en``
            and ``whisper_basic`` are accepted so a config can name the
            cleaner exactly as the recipe does.
        **kwargs: Passed to ``BasicTextNormalizer``; it takes
            ``remove_diacritics`` and ``split_letters``. Note that
            ``split_letters=True`` needs the ``regex`` package, which is not
            a SPLET dependency, and that egs2 uses neither.

    Returns:
        A callable taking one string and returning the normalized string.

    Raises:
        ValueError: If the language is not one this normalizer covers.
    """
    if language not in _FLAVOURS:
        raise ValueError(
            f"unknown whisper normalizer '{language}'. Available: "
            f"{sorted(_FLAVOURS)}. Whisper ships an English normalizer and a "
            "language-agnostic basic one; there is no per-language variant to "
            "select, and inventing one would not match any published result."
        )
    if language in ("basic", "whisper_basic"):
        return BasicTextNormalizer(**kwargs)
    if kwargs:
        raise ValueError(
            f"EnglishTextNormalizer takes no options; got {sorted(kwargs)}. "
            "remove_diacritics and split_letters belong to language='basic'."
        )
    return EnglishTextNormalizer()

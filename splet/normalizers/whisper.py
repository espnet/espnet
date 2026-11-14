#!/usr/bin/env python3

# Copyright 2026 ESPnet Developers
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Whisper-style normalization, as ESPnet recipes apply it at test time.

``egs2`` reaches this through ``espnet2.text.cleaner``, whose ``whisper_en``
and ``whisper_basic`` cleaners construct OpenAI's ``EnglishTextNormalizer``
and ``BasicTextNormalizer`` with their default arguments. The two variants
here are those two, under the same names.

It is a dependency, not a copy. ``whisper-normalizer`` on PyPI is a
redistribution of the normalizers from ``openai-whisper``, and it was
checked against them rather than assumed: both normalizers over the
fixtures plus a set of strings built to hit the spelling table, the number
formatter, the contraction expander and the currency handler, with no
differences. Vendoring instead would have been 2534 lines to keep in step
with a file that moves, and ``openai-whisper`` itself is not an option --
it brings torch and tiktoken into a package whose whole point is that it
can be lifted out of ESPnet.

The import is guarded and the dependency lives in the ``splet`` extra,
which ``ci/install.sh`` does not install: it installs ``.[test,all]``, and
``all`` does not include ``splet``. So this must fail with an instruction
rather than an ImportError traceback, and the tests that need it skip.
"""

from __future__ import annotations

from typing import Callable

try:
    from whisper_normalizer.basic import BasicTextNormalizer
    from whisper_normalizer.english import EnglishTextNormalizer
except ImportError:  # pragma: no cover - exercised by the no-extras CI job
    BasicTextNormalizer = None
    EnglishTextNormalizer = None

#: Variant -> the espnet2 cleaner name it reproduces.
VARIANT_CHOICES = {"en": "whisper_en", "basic": "whisper_basic"}


def whisper_setup(
    variant: str = "en", split_letters: bool = False
) -> Callable[[str], str]:
    """Build a Whisper text normalizer.

    Args:
        variant: ``en`` for ``EnglishTextNormalizer``, which is what the
            ``whisper_en`` cleaner uses; ``basic`` for
            ``BasicTextNormalizer``, which is ``whisper_basic`` and what
            Whisper itself applies to every other language.
        split_letters: Passed to ``BasicTextNormalizer``, which inserts a
            space between characters when it is set. ``espnet2`` constructs
            the normalizer with the default, so leave this alone to
            reproduce a recipe.

    Returns:
        The normalizer, a ``str -> str`` callable.

    Raises:
        ValueError: If ``variant`` is not one of :data:`VARIANT_CHOICES`,
            or if ``split_letters`` is set for the English variant, which
            does not take it.
        ImportError: If whisper-normalizer is not installed.
    """
    if variant not in VARIANT_CHOICES:
        raise ValueError(
            f"unknown whisper variant '{variant}'; "
            f"expected one of {sorted(VARIANT_CHOICES)}"
        )
    if EnglishTextNormalizer is None:
        raise ImportError(
            "Whisper-style normalization requires whisper-normalizer: "
            "pip install whisper-normalizer, or install espnet[splet]."
        )
    if variant == "en":
        if split_letters:
            raise ValueError(
                "split_letters applies to variant='basic' only; "
                "EnglishTextNormalizer does not take it"
            )
        normalizer = EnglishTextNormalizer()
    else:
        normalizer = BasicTextNormalizer(split_letters=split_letters)

    # Call it rather than returning it: the normalizer objects are callable
    # but not plain functions, and the pipeline's config echo reads better
    # with a uniform signature.
    return lambda text: normalizer(text)


__all__ = ["VARIANT_CHOICES", "whisper_setup"]

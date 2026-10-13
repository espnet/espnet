#!/usr/bin/env python3

# Copyright 2026 ESPnet Developers
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Whisper-style normalization.

Not implemented yet. It is listed in the first milestone of
espnet/espnet#6760 and it is the single normalizer most likely to be
reached for, so it gets a module of its own rather than an entry in a TODO
list -- and a failure that says what is missing rather than a quiet
approximation that changes every number it touches.

What it has to do, and why it cannot be approximated: Whisper's
``EnglishTextNormalizer`` applies a spelling table, a number formatter, a
contraction expander and a currency/unit handler, in a fixed order. Each of
those moves WER by a point or more on its own. Reimplementing three of the
four and calling the result "whisper-style" produces a number that cannot be
compared with any published Whisper result, which is exactly the failure this
toolkit exists to prevent.
"""

from __future__ import annotations

from typing import Callable

_MESSAGE = (
    "Whisper-style normalization is not implemented yet (espnet/espnet#6760). "
    "It must reproduce openai-whisper's EnglishTextNormalizer, validated "
    "against it on a fixed set of strings, before it can be used to report a "
    "number. Until then, compose the steps you actually want from "
    "splet.normalizers.basic and state them in the config."
)


def whisper_setup(language: str = "en") -> Callable[[str], str]:
    """Raise: Whisper-style normalization is not implemented yet."""
    raise NotImplementedError(_MESSAGE)

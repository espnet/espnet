# Copyright (c) 2022 OpenAI
# Licensed under the MIT License; see LICENSE in this directory.
# Vendored from openai-whisper 20250625, whisper/normalizers/basic.py
# For details, see: https://github.com/openai/whisper/blob/main/LICENSE
#
# This file is MIT, not Apache 2.0 like the rest of ESPnet.

"""Vendored from openai-whisper 20250625, whisper/normalizers/basic.py.

MIT licensed, see LICENSE in this directory. Copied rather than imported
because ``import whisper`` pulls in torch, which splet/ may not depend on
(ci/check_splet_independence.py), and copied rather than reimplemented
because an approximation of this produces numbers that cannot be compared
with any published Whisper result.

Changed from upstream: ``import regex`` is deferred into the one branch that
uses it, so the default path needs no third-party package. Nothing else.
"""

import re
import unicodedata

# non-ASCII letters that are not separated by "NFKD" normalization
ADDITIONAL_DIACRITICS = {
    "œ": "oe",
    "Œ": "OE",
    "ø": "o",
    "Ø": "O",
    "æ": "ae",
    "Æ": "AE",
    "ß": "ss",
    "ẞ": "SS",
    "đ": "d",
    "Đ": "D",
    "ð": "d",
    "Ð": "D",
    "þ": "th",
    "Þ": "th",
    "ł": "l",
    "Ł": "L",
}


def remove_symbols_and_diacritics(s: str, keep=""):
    """
    Replace any other markers, symbols, and punctuations with a space,
    and drop any diacritics (category 'Mn' and some manual mappings)
    """
    return "".join(
        (
            c
            if c in keep
            else (
                ADDITIONAL_DIACRITICS[c]
                if c in ADDITIONAL_DIACRITICS
                else (
                    ""
                    if unicodedata.category(c) == "Mn"
                    else " " if unicodedata.category(c)[0] in "MSP" else c
                )
            )
        )
        for c in unicodedata.normalize("NFKD", s)
    )


def remove_symbols(s: str):
    """
    Replace any other markers, symbols, punctuations with a space, keeping diacritics
    """
    return "".join(
        " " if unicodedata.category(c)[0] in "MSP" else c
        for c in unicodedata.normalize("NFKC", s)
    )


class BasicTextNormalizer:
    def __init__(self, remove_diacritics: bool = False, split_letters: bool = False):
        self.clean = (
            remove_symbols_and_diacritics if remove_diacritics else remove_symbols
        )
        self.split_letters = split_letters

    def __call__(self, s: str):
        s = s.lower()
        s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)  # remove words between brackets
        s = re.sub(r"\(([^)]+?)\)", "", s)  # remove words between parenthesis
        s = self.clean(s).lower()

        if self.split_letters:
            # Deferred and guarded: \X (a grapheme cluster) is a regex-module
            # feature that re does not have, this branch is off by default,
            # and no egs2 recipe turns it on. Keeping the import here is what
            # lets the default path run with no third-party package at all.
            try:
                import regex
            except ImportError as error:  # pragma: no cover - needs regex absent
                raise ImportError(
                    "split_letters=True needs the regex package, which SPLET "
                    "does not require otherwise: pip install regex"
                ) from error

            s = " ".join(regex.findall(r"\X", s, regex.U))

        s = re.sub(
            r"\s+", " ", s
        )  # replace any successive whitespace characters with a space

        return s

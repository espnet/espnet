"""Turn an OWSM text stream into something scorable."""

from __future__ import annotations

import re

# <eng>, <asr>, <st_deu>, <0.00>, <notimestamps>, <na>, <nospeech> -- every
# OWSM tag has this shape, and none of them is scorable text.
_TAG = re.compile(r"<[^<>]*>")


def strip_markup(text: str) -> str:
    """Drop every tag and collapse the whitespace they leave behind."""
    return " ".join(_TAG.sub(" ", text).split())

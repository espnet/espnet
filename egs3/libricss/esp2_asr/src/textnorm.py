"""Scoring text normalization for LibriCSS.

Port of egs/libri_css/asr1/local/wer_output_filter: lowercase everything,
drop bracketed non-speech tokens such as ``[laugh]``, and map the hesitation
spelling variants mhm/mm/mmm to hmm. Applied identically to references and
hypotheses before WER computation.
"""

from __future__ import annotations

import re

# egs1: `unless $a =~ /\[.*\]/` - any token containing a bracketed span.
_BRACKET_RE = re.compile(r"\[.*\]")

# egs1: sed 's/\<mhm\>/hmm/g; s/\<mm\>/hmm/g; s/\<mmm\>/hmm/g'
_HESITATIONS = {"mhm": "hmm", "mm": "hmm", "mmm": "hmm"}


def normalize_text(text: str) -> str:
    """Normalize one transcript line for scoring.

    Args:
        text: Raw transcript (reference or hypothesis).

    Returns:
        Lowercased, hesitation-folded text with bracketed tokens removed.

    Example:
        >>> normalize_text("HELLO [laugh] MM WORLD")
        'hello hmm world'
    """
    tokens = []
    for tok in str(text).strip().split():
        if _BRACKET_RE.search(tok):
            continue
        tok = tok.lower()
        tokens.append(_HESITATIONS.get(tok, tok))
    return " ".join(tokens)

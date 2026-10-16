"""Shared helpers for the LibriCSS metrics.

Word-level error counting mirrors Kaldi's ``compute-wer --mode=present`` as
used by egs/libri_css/asr1: present mode simply aligns the two token
sequences, which is exactly jiwer's default word error rate computation.
Empty-string cases (which make jiwer's alignment degenerate and produced NaN
costs in egs1) are handled analytically instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple


def _jiwer_counts(ref: str, hyp: str) -> Tuple[int, int, int]:
    """Return (insertions, deletions, substitutions) across jiwer versions."""
    try:
        import jiwer  # noqa: PLC0415
    except ImportError as e:
        raise RuntimeError(
            "LibriCSS metrics require jiwer: `pip install jiwer` " "(or espnet[asr])."
        ) from e
    if hasattr(jiwer, "process_words"):
        out = jiwer.process_words(ref, hyp)
        if isinstance(out, dict):  # jiwer 2.x returned a dict
            return (
                int(out["insertions"]),
                int(out["deletions"]),
                int(out["substitutions"]),
            )
        return int(out.insertions), int(out.deletions), int(out.substitutions)
    out = jiwer.compute_measures(ref, hyp)  # jiwer < 2.3
    return (
        int(out["insertions"]),
        int(out["deletions"]),
        int(out["substitutions"]),
    )


def pair_word_counts(ref_text: str, hyp_text: str) -> Tuple[int, int, int, int, int]:
    """Word-level counts for one reference/hypothesis pair.

    Args:
        ref_text: Normalized reference text.
        hyp_text: Normalized hypothesis text.

    Returns:
        ``(ins, del_, sub, ref_wc, hyp_wc)``: insertion, deletion and
        substitution counts plus the reference and hypothesis word counts.
        Empty inputs are handled analytically: an empty reference turns all
        hypothesis words into insertions (ref_wc = 0), and an empty
        hypothesis turns all reference words into deletions.
    """
    ref = str(ref_text).strip()
    hyp = str(hyp_text).strip()
    ref_wc = len(ref.split())
    hyp_wc = len(hyp.split())
    if ref_wc == 0 and hyp_wc == 0:
        return 0, 0, 0, 0, 0
    if ref_wc == 0:
        return hyp_wc, 0, 0, 0, hyp_wc
    if hyp_wc == 0:
        return 0, ref_wc, 0, ref_wc, 0
    ins, del_, sub = _jiwer_counts(ref, hyp)
    return ins, del_, sub, ref_wc, hyp_wc


def read_scp(path: Path) -> Dict[str, str]:
    """Read a two-column scp file into a dict."""
    entries: Dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            entries[parts[0]] = parts[1] if len(parts) > 1 else ""
    return entries

#!/usr/bin/env python3
"""Align reference/hypothesis Kaldi-style text files for ST scoring.

Strips OWSM prompt tokens (e.g. ``<eng><st_deu><notimestamps>``) from the
reference, keeps only utterance IDs present in both files, and writes the
aligned plain-text reference and hypothesis files consumed by ``sacrebleu``.
"""

from __future__ import annotations

import argparse
import re
import sys
from typing import Dict


def read_kaldi_text(path: str, strip_prompt: bool = False) -> Dict[str, str]:
    data: Dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(maxsplit=1)
            utt = parts[0]
            text = parts[1] if len(parts) == 2 else ""
            if strip_prompt:
                text = re.sub(r"^(?:<[^>]+>)+\s*", "", text).strip()
            data[utt] = text
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ref_in", help="Kaldi-style reference text (with prompt)")
    parser.add_argument("hyp_in", help="Kaldi-style hypothesis text")
    parser.add_argument("ref_out", help="Aligned reference output (one line per utt)")
    parser.add_argument("hyp_out", help="Aligned hypothesis output (one line per utt)")
    args = parser.parse_args()

    ref = read_kaldi_text(args.ref_in, strip_prompt=True)
    hyp = read_kaldi_text(args.hyp_in)
    keys = sorted(set(ref) & set(hyp))

    missing_ref = len(set(hyp) - set(ref))
    missing_hyp = len(set(ref) - set(hyp))
    if missing_ref or missing_hyp:
        print(
            f"Aligned {len(keys)} utterances "
            f"({missing_ref} hyp-only, {missing_hyp} ref-only)",
            file=sys.stderr,
        )

    with (
        open(args.ref_out, "w", encoding="utf-8") as f_ref,
        open(args.hyp_out, "w", encoding="utf-8") as f_hyp,
    ):
        for key in keys:
            f_ref.write(ref[key] + "\n")
            f_hyp.write(hyp[key] + "\n")


if __name__ == "__main__":
    main()

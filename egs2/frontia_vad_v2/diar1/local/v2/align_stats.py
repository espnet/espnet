#!/usr/bin/env python3
"""Distributions behind the alignment settings (--min_word_score, --min_pause).

    python3 local/v2/align_stats.py dump/v2/librispeech

Reads align.tsv (every word span and score, also of dropped recordings) and
manifest.tsv of the aligned dumps under the given directories, and prints:
word score percentiles; the share of recordings a min_word_score would drop;
inter-word gap percentiles and the share of gaps a min_pause would make
non-speech; and the speech fraction kept.
"""

import collections
import sys
from pathlib import Path

import numpy as np


def main():
    words = collections.defaultdict(list)
    for root in sys.argv[1:]:
        for a in sorted(Path(root).glob("**/align.tsv")):
            for line in a.read_text().splitlines()[1:]:
                rid, _, s, e, c = line.split("\t")[:5]
                words[rid].append((float(s), float(e), float(c)))
    if not words:
        sys.exit("no align.tsv")
    scores = np.array([c for v in words.values() for _, _, c in v])
    worst = np.array([min(c for _, _, c in v) for v in words.values()])
    gaps = np.array([b[0] - a[1] for v in words.values() for a, b in zip(v, v[1:])])
    print(f"recordings {len(words)}, words {len(scores)}")
    pct = (0.1, 1, 5, 25, 50)
    print(
        "word score percentiles",
        dict(zip(pct, np.round(np.percentile(scores, pct), 2))),
    )
    for th in (-5, -10, -15, -20, -30):
        share = np.mean(worst < th) * 100
        print(f"min_word_score {th}: drops {share:.2f} % of recordings")
    pct = (50, 90, 95, 99)
    print(
        "inter-word gap percentiles (s)",
        dict(zip(pct, np.round(np.percentile(gaps, pct), 3))),
    )
    for mp in (0.1, 0.2, 0.3, 0.5):
        print(
            f"min_pause {mp}: {np.mean(gaps >= mp) * 100:.2f} % of gaps are non-speech"
        )
    total = speech = 0.0
    for root in sys.argv[1:]:
        for m in sorted(Path(root).glob("**/manifest.tsv")):
            if m.parent.name == "before_align":
                continue
            for line in m.read_text().splitlines()[1:]:
                f = line.split("\t")
                total += float(f[9])
                speech += float(f[10])
    if total:
        print(f"kept: {total / 3600:.2f} h, speech {speech / total * 100:.1f} %")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare two diarize.scp outputs of diar_inference recording by recording.

Exits with an error when a recording is missing from one of them, or when its
predictions differ in shape or by more than the tolerance.

Usage: compare_predictions.py <diarize.scp> <diarize.scp> [--atol 1e-5]
"""

import argparse
import sys

import numpy as np


def read_scp(path):
    with open(path, encoding="utf-8") as f:
        return dict(line.split(maxsplit=1) for line in f if line.strip())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--atol", type=float, default=1e-5)
    args = ap.parse_args()

    a, b = read_scp(args.a), read_scp(args.b)
    if a.keys() != b.keys():
        sys.exit(f"different recordings: {sorted(a.keys() ^ b.keys())}")
    worst = 0.0
    for key in sorted(a):
        x, y = np.load(a[key].strip()), np.load(b[key].strip())
        if x.shape != y.shape:
            sys.exit(f"{key}: shape {x.shape} != {y.shape}")
        # An EEND-EDA model can find no speaker: (frames, 0) arrays, nothing to
        # compare beyond the shape.
        if x.size:
            worst = max(worst, float(np.abs(x - y).max()))
    if worst > args.atol:
        sys.exit(f"predictions differ by up to {worst:.2e} (atol {args.atol})")
    print(
        f"{len(a)} recordings: same shapes, predictions differ by at most {worst:.2e}"
    )


if __name__ == "__main__":
    main()

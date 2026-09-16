#!/usr/bin/env python3
"""Read ci/image_variants.json for the workflows.

pairs   one "<python> <pytorch>" per line, for shell loops
matrix  the grid as compact JSON, for strategy.matrix via fromJSON

matrix takes --newest-pytorch, which keeps every python but only the highest
pytorch. It exists for the integration grid on pull requests: 14 recipe tasks
across the full 2x3 grid is 84 jobs against a 20-wide cap, five waves, and it
is the whole critical path of the workflow. Measured over 300 runs, no
integration failure was ever specific to a pytorch version - the ones that were
version-specific were specific to a python, and failed on all three pytorches.
Pushes to master still run the full grid, so the pytorch axis is not dropped,
only moved off the path that gates a review.
"""

import json
import sys
from pathlib import Path

KEYS = ("python-version", "pytorch-version")
NEWEST = "--newest-pytorch"
PATH = Path(__file__).resolve().parent / "image_variants.json"


def grid() -> dict:
    data = json.loads(PATH.read_text())
    return {key: data[key] for key in KEYS}


def newest(versions: list) -> str:
    """The highest version, by number rather than by position in the file."""
    return max(versions, key=lambda v: tuple(int(part) for part in v.split(".")))


def main() -> int:
    what = sys.argv[1] if len(sys.argv) > 1 else "matrix"
    arguments = sys.argv[2:]
    variants = grid()
    if what == "matrix":
        if NEWEST in arguments:
            arguments = [a for a in arguments if a != NEWEST]
            variants["pytorch-version"] = [newest(variants["pytorch-version"])]
        for extra in arguments:
            key, _, values = extra.partition("=")
            if not key or not values:
                sys.exit(f"expected key=a,b,c, got: {extra}")
            variants[key] = values.split(",")
        print(json.dumps(variants, separators=(",", ":")))
    elif what == "pairs":
        for python in variants["python-version"]:
            for pytorch in variants["pytorch-version"]:
                print(python, pytorch)
    else:
        sys.exit(f"unknown mode: {what}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

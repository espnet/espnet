#!/usr/bin/env python3
"""Read ci/image_variants.json for the workflows.

  pairs   one "<python> <pytorch>" per line, for shell loops
  matrix  the grid as compact JSON, for strategy.matrix via fromJSON
"""

import json
import sys
from pathlib import Path

KEYS = ("python-version", "pytorch-version")
PATH = Path(__file__).resolve().parent / "image_variants.json"


def grid() -> dict:
    data = json.loads(PATH.read_text())
    return {key: data[key] for key in KEYS}


def main() -> int:
    what = sys.argv[1] if len(sys.argv) > 1 else "matrix"
    variants = grid()
    if what == "matrix":
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

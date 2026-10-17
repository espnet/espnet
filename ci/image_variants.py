#!/usr/bin/env python3
"""Read ci/image_variants.json for the workflows.

pairs   one "<python> <pytorch>" per line, for shell loops
matrix  the grid as compact JSON, for strategy.matrix via fromJSON

matrix takes --newest-pytorch, which keeps every python but only one pytorch.
It exists for the integration grid on pull requests: 14 recipe tasks across the
full 2x3 grid is 84 jobs against a 20-wide cap, five waves, and the whole
critical path of the workflow. Measured over 300 runs, no integration failure
was ever specific to a pytorch version - the ones that were version-specific
were specific to a python and failed on all three pytorches. Pushes to master
still run the full grid, so the axis is not dropped, only moved off the path
that gates a review.

The one it keeps is the highest that the whole suite actually supports. A
version listed in install_k2.sh's k2_missing_for is skipped, because those are
exactly the versions where parts of the suite quietly skip themselves: picking
the newest blindly would have put every pull request on torch 2.14.0, where the
k2 blocks in ci/test_integration_espnet2.sh never run at all.
"""

import json
import re
import sys
from pathlib import Path

KEYS = ("python-version", "pytorch-version")
NEWEST = "--newest-pytorch"
INSTALL_K2 = Path(__file__).resolve().parent.parent / "tools/installers/install_k2.sh"
PATH = Path(__file__).resolve().parent / "image_variants.json"


def grid() -> dict:
    data = json.loads(PATH.read_text())
    return {key: data[key] for key in KEYS}


def _order(version: str) -> tuple:
    return tuple(int(part) for part in version.split("."))


def incomplete() -> set:
    """Torch versions the grid builds but k2 publishes no wheel for."""
    try:
        text = INSTALL_K2.read_text()
    except OSError:
        return set()
    match = re.search(r'^k2_missing_for="([^"]*)"', text, re.M)
    return set(match.group(1).split()) if match else set()


def newest(versions: list) -> str:
    """The highest version the suite fully supports.

    By number rather than by position in the file - 2.9.1 against 2.10.0 is
    the pair a string sort gets wrong - and skipping the versions k2 has no
    wheel for, since on those the k2 parts of the suite skip themselves and a
    grid narrowed onto one would stop running them anywhere but master.
    """
    complete = [v for v in versions if v not in incomplete()]
    return max(complete or versions, key=_order)


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

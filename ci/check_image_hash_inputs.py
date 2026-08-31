#!/usr/bin/env python3
"""Assert the two image-tag hash input lists are identical.

The tag of the prebuilt CI image is a hash of the files that determine its
contents. That list appears twice - once where the image is built, once where a
job resolves the tag to pull - and the two must agree exactly. If they drift,
the build publishes one tag and every job asks for another, so nothing can be
pulled.

That is not hypothetical: adding ci/install_kaldi.sh to one list and not the
other is what this check was written for.
"""

import re
import sys
from pathlib import Path

BUILD = Path(".github/workflows/build_ci_image.yml")
CONSUMER = Path(".github/workflows/ci_on_ubuntu.yml")
PATTERN = re.compile(r"hashFiles\(\s*('ci/install\.sh'[^)]*)\)")


def inputs(path: Path) -> list:
    match = PATTERN.search(path.read_text())
    if match is None:
        sys.exit(f"{path}: no image-tag hashFiles(...) call found")
    return [item.strip().strip("'\"") for item in match.group(1).split(",")]


def main() -> int:
    build, consumer = inputs(BUILD), inputs(CONSUMER)
    if build == consumer:
        print(f"image-tag hash inputs agree ({len(build)} entries)")
        return 0
    print("The image-tag hash inputs disagree.", file=sys.stderr)
    print(f"  {BUILD}:\n    {build}", file=sys.stderr)
    print(f"  {CONSUMER}:\n    {consumer}", file=sys.stderr)
    print(
        "\nBoth lists must name the same files in the same order, or the image "
        "is published under one tag and requested under another.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())

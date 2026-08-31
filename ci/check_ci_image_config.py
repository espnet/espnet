#!/usr/bin/env python3
"""Check that the prebuilt CI image configuration is self-consistent.

The tag of the prebuilt CI image is a hash of the files that determine its
contents. That list appears twice - once where the image is built, once where a
job resolves the tag to pull - and the two must agree exactly. If they drift,
the build publishes one tag and every job asks for another, so nothing can be
pulled.

That is not hypothetical: adding ci/install_kaldi.sh to one list and not the
other is what this check was written for.

Second, every python x pytorch combination a job asks for must be one that
ci/image_variants.json says is built. A job asking for a variant nobody builds
gets `manifest unknown`, and the fallback would quietly hide it behind a slow
build instead.
"""

import json
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


def variants() -> tuple:
    data = json.loads(Path("ci/image_variants.json").read_text())
    return tuple(data["python-version"]), tuple(data["pytorch-version"])


def _items(group: str) -> list:
    return [item.strip().strip("\"'") for item in group.split(",")]


def job_matrices() -> dict:
    """Literal python/pytorch lists still written out in the workflow."""
    text = CONSUMER.read_text()
    found = {}
    for job in re.finditer(r"^  ([a-z_0-9]+):$", text, flags=re.M):
        name = job.group(1)
        block = text[job.end() :]
        end = re.search(r"^  [a-z_0-9]+:$", block, flags=re.M)
        block = block[: end.start()] if end else block
        py = re.search(r"^        python-version: \[(.*?)\]$", block, flags=re.M)
        th = re.search(r"^        pytorch-version: \[(.*?)\]$", block, flags=re.M)
        if py and th:
            found[name] = (_items(py.group(1)), _items(th.group(1)))
    return found


def check_variants() -> list:
    pythons, pytorches = variants()
    problems = []
    for job, (job_py, job_th) in job_matrices().items():
        for value in job_py:
            if value not in pythons:
                problems.append(f"{job}: python {value} is not a built variant")
        for value in job_th:
            if value not in pytorches:
                problems.append(f"{job}: pytorch {value} is not a built variant")
    return problems


def main() -> int:
    bad = check_variants()
    for problem in bad:
        print(problem, file=sys.stderr)
    build, consumer = inputs(BUILD), inputs(CONSUMER)
    if build == consumer and not bad:
        print(
            f"hash inputs agree ({len(build)} entries); "
            "every job matrix is a built variant"
        )
        return 0
    if bad:
        return 1
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

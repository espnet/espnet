#!/usr/bin/env python3
"""Decide whether a pull request can reach one of the integration suites.

    ci/integration_is_relevant.py <espnet2|espnet3> < changed-paths

Reads changed paths on stdin, one per line. Exits 0 if any of them could
affect what that suite runs, 1 if none could.

The lists are deliberately broad. Being wrong in the "skip it" direction means
a pull request merges without the recipe tests having run, and the only thing
that would then catch it is the master push - after the merge. Being wrong in
the "run it" direction costs jobs. Those are not symmetric, so anything that
might reach a mini_an4 run is in.

espnet3 carries espnet2/ for a reason that is easy to miss: espnet3 imports
espnet2 throughout - 51 references to espnet2.asr alone - so a change to
espnet2 can break the espnet3 recipes without touching espnet3 at all.
"""

import sys

# What both suites run through: the environment, the packaging, the harness.
SHARED = (
    "espnet2/",  # espnet3 imports it too; see the note above
    "utils/",
    "test_utils/",
    "tools/",
    "pyproject.toml",
    "setup.cfg",
    "ci/install",  # install.sh, install_kaldi.sh
    "ci/no_redistribute.txt",
    "ci/integration_is_relevant.py",  # these lists
    ".github/workflows/ci_on_ubuntu.yml",  # the jobs that run them
    ".github/actions/",  # and the composite actions they use
)

RELEVANT = {
    "espnet2": SHARED
    + (
        "egs2/TEMPLATE/",  # the recipe scripts mini_an4 symlinks to
        "egs2/mini_an4/",  # the recipes themselves
        "ci/test_integration_espnet2.sh",
    ),
    "espnet3": SHARED
    + (
        "espnet3/",
        "egs3/",  # mini_an4, TEMPLATE and integration_test are all small
        "ci/test_integration_espnet3.sh",
        "ci/test_integration_espnet3_publication.sh",
    ),
}


def relevant(suite: str, paths) -> list:
    prefixes = RELEVANT[suite]
    return [p for p in paths if any(p == r or p.startswith(r) for r in prefixes)]


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in RELEVANT:
        sys.exit(f"usage: {sys.argv[0]} <{'|'.join(RELEVANT)}>")
    suite = sys.argv[1]
    paths = [line.strip() for line in sys.stdin if line.strip()]
    if not paths:
        # No list means no information, and no information is not evidence of
        # irrelevance. Run them.
        print(f"no changed paths were given; running the {suite} integration tests")
        return 0
    hits = relevant(suite, paths)
    if hits:
        print(
            f"{len(hits)} of {len(paths)} changed paths can reach the {suite} "
            f"integration tests, for example: {', '.join(hits[:3])}"
        )
        return 0
    print(
        f"none of the {len(paths)} changed paths can reach the {suite} "
        "integration tests"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())

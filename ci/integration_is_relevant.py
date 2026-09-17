#!/usr/bin/env python3
"""Decide whether a pull request can reach the espnet2 integration tests.

Reads changed paths on stdin, one per line. Exits 0 if any of them could
affect what ci/test_integration_espnet2.sh runs, 1 if none could.

The list is deliberately broad. Being wrong in the "skip it" direction means a
pull request merges without the recipe tests having run, and the only thing
that would then catch it is the master push - after the merge. Being wrong in
the "run it" direction costs 28 jobs. Those are not symmetric, so anything
that might reach a mini_an4 run is in the list.

What the script actually runs is egs2/mini_an4/<task>/run.sh, which are
symlinks into egs2/TEMPLATE, calling espnet2/bin through utils/ and the
environment tools/ builds.
"""

import sys

RELEVANT = (
    "espnet2/",  # the code under test
    "egs2/TEMPLATE/",  # the recipe scripts mini_an4 symlinks to
    "egs2/mini_an4/",  # the recipes themselves
    "utils/",  # the shell utilities those call
    "test_utils/",
    "tools/",  # the environment they run in
    "pyproject.toml",
    "setup.cfg",
    "ci/install",  # install.sh, install_kaldi.sh
    "ci/no_redistribute.txt",
    "ci/test_integration_espnet2.sh",  # the script itself
    "ci/integration_is_relevant.py",  # and this list
    ".github/workflows/ci_on_ubuntu.yml",  # the job that runs it
    ".github/actions/",  # and the composite actions it uses
)


def relevant(paths) -> list:
    return [p for p in paths if any(p == r or p.startswith(r) for r in RELEVANT)]


def main() -> int:
    paths = [line.strip() for line in sys.stdin if line.strip()]
    if not paths:
        # No list means no information, and no information is not evidence of
        # irrelevance. Run them.
        print("no changed paths were given; running the integration tests")
        return 0
    hits = relevant(paths)
    if hits:
        print(f"{len(hits)} of {len(paths)} changed paths can reach the "
              f"integration tests, for example: {', '.join(hits[:3])}")
        return 0
    print(f"none of the {len(paths)} changed paths can reach the integration "
          "tests")
    return 1


if __name__ == "__main__":
    sys.exit(main())

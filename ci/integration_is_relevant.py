#!/usr/bin/env python3
"""Decide whether a pull request can reach one of the integration suites.

    gh api ... --jq '.[] | {filename, previous_filename}' |
        ci/integration_is_relevant.py <espnet2|espnet3>

Reads the pull request's changed files as JSON lines on stdin. Exits 0 if any
of them could affect what that suite runs, 1 if none could.

Both paths of a rename count. The API reports the destination in `filename`
and the source in `previous_filename`, and moving espnet2/foo.py somewhere
unrelated changes espnet2 while leaving no espnet2 path in `filename` - so
reading only that would skip the tests for the change most likely to need
them. Renames are not rare here: two of the last forty pull requests had
them.

The files endpoint returns at most 3000 records even when paginated, and says
so in no way the caller can see. A pull request larger than that could
present only unrelated paths, so hitting the cap runs everything.

The lists are deliberately broad. Being wrong in the "skip it" direction means
a pull request merges without the recipe tests having run, and the only thing
that would then catch it is the master push - after the merge. Being wrong in
the "run it" direction costs jobs. Those are not symmetric, so anything that
might reach a mini_an4 run is in.

espnet3 carries espnet2/ for a reason that is easy to miss: espnet3 imports
espnet2 throughout - 51 references to espnet2.asr alone - so a change to
espnet2 can break the espnet3 recipes without touching espnet3 at all.
"""

import json
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


# What GET /repos/{owner}/{repo}/pulls/{number}/files will return and no more.
API_FILE_CAP = 3000


def read(stream) -> tuple:
    """(paths, record count), or (None, n) if a line could not be read."""
    paths, records = [], 0
    for line in stream:
        line = line.strip()
        if not line:
            continue
        records += 1
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            return None, records
        if not isinstance(entry, dict) or "filename" not in entry:
            return None, records
        paths.append(entry["filename"])
        if entry.get("previous_filename"):
            paths.append(entry["previous_filename"])
    return paths, records


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in RELEVANT:
        sys.exit(f"usage: {sys.argv[0]} <{'|'.join(RELEVANT)}>")
    suite = sys.argv[1]
    paths, records = read(sys.stdin)
    if paths is None:
        print(
            f"could not read the changed files; running the {suite} "
            "integration tests"
        )
        return 0
    if records >= API_FILE_CAP:
        print(
            f"{records} changed files reaches the API's {API_FILE_CAP}-record "
            f"cap, so the list may be short; running the {suite} integration "
            "tests"
        )
        return 0
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

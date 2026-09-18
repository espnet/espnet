#!/usr/bin/env python3
"""Report the repository's demo links that no longer work.

    python3 ci/check_demo_links.py              # scan, print a report
    python3 ci/check_demo_links.py --self-check # offline test of the scanner

Two kinds of link rot were found by hand on 2026-09-18, both invisible to a
plain link checker because the pages answer 200:

  * a notebook that moved when espnet/notebook was reorganised - Colab shows
    "Notebook not found" rather than failing the request;
  * a Hugging Face Space whose container is dead - the page renders, with a
    runtime error inside it. Four of the five Spaces the README linked had
    been erroring since 2021-2022.

So each kind is checked against an API instead: the notebook against the
repository's file list, the Space against its runtime stage.

Exit status: 0 nothing broken, 1 something broken, 2 the scan itself failed
(no network, API error) - which must not be read as "nothing broken".
"""

import argparse
import errno
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from typing import Dict, List, Set, Tuple

NOTEBOOK_REPO = "espnet/notebook"
# Colab and GitHub spellings of a path inside the notebook repository.
NOTEBOOK_LINK = re.compile(
    r"https://(?:colab\.research\.google\.com/github|github\.com)"
    r"/espnet/notebook/blob/(?P<ref>[^/]+)/(?P<path>[^ )\"'>]+\.ipynb)"
)
# Only a link to the app itself. A deeper path (/tree/main, /blob/...) browses
# the Space's files, which is how the repository cites code it adapted; those
# pages work whether or not the container runs.
SPACE_LINK = re.compile(
    r"https://huggingface\.co/spaces/(?P<owner>[A-Za-z0-9_.-]+)"
    r"/(?P<name>[A-Za-z0-9_.-]+)(?![A-Za-z0-9_.-]|/\S)"
)
# A Space is usable when it is running; a paused or sleeping one wakes up on a
# visit, so only these stages are reported.
BROKEN_STAGES = {"RUNTIME_ERROR", "BUILD_ERROR", "CONFIG_ERROR", "DELETED"}
TIMEOUT = 30


class ScanError(RuntimeError):
    """The scan could not be completed, which is not the same as a broken link."""


def documentation_files() -> List[str]:
    """Tracked files that point people at a demo: docs and the shell scripts.

    utils/synth_wav.sh sends beginners to a Colab notebook in its help text,
    and that link had rotted with the rest.
    """
    try:
        out = subprocess.run(
            ["git", "ls-files", "*.md", "*.sh"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as e:
        # not "nothing is broken": the scan never looked
        raise ScanError(f"cannot list tracked documentation files: {e}") from e
    return [p for p in out.stdout.split("\n") if p]


def find_links(
    texts: Dict[str, str],
) -> Tuple[Dict[Tuple[str, str], Set[str]], Dict[str, Set[str]]]:
    """Return {(ref, notebook path): {files}} and {space id: {files}}."""
    notebooks: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    spaces: Dict[str, Set[str]] = defaultdict(set)
    for name, text in texts.items():
        for m in NOTEBOOK_LINK.finditer(text):
            notebooks[(m.group("ref"), m.group("path"))].add(name)
        for m in SPACE_LINK.finditer(text):
            spaces[f"{m.group('owner')}/{m.group('name')}"].add(name)
    return dict(notebooks), dict(spaces)


def _get_json(url: str):
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise ScanError(f"{url}: HTTP {e.code}") from e
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        raise ScanError(f"{url}: {e}") from e


def notebook_paths(ref: str) -> Set[str]:
    """The files the notebook repository holds at ``ref`` (branch, tag or sha)."""
    tree = _get_json(
        f"https://api.github.com/repos/{NOTEBOOK_REPO}/git/trees/{ref}?recursive=1"
    )
    if tree is None or "tree" not in tree:
        raise ScanError(f"cannot list {NOTEBOOK_REPO} at {ref}")
    if tree.get("truncated"):
        raise ScanError(f"{NOTEBOOK_REPO} file list at {ref} came back truncated")
    return {t["path"] for t in tree["tree"]}


def space_stage(space_id: str) -> str:
    d = _get_json(f"https://huggingface.co/api/spaces/{space_id}")
    if d is None:
        return "DELETED"
    return (d.get("runtime") or {}).get("stage") or "UNKNOWN"


def scan() -> List[str]:
    texts = {}
    for name in documentation_files():
        if os.path.islink(name):
            try:
                os.stat(name)  # follows the link
            except OSError as e:
                # three tracked scripts under egs2 are symlinks whose target is
                # not in the repository; they hold no text to scan here. Any
                # other error is a file this scan failed to read, not an empty
                # one - os.path.exists() would have called both of them False.
                if e.errno not in (errno.ENOENT, errno.ENOTDIR):
                    raise ScanError(f"cannot read {name}: {e}") from e
                continue
        try:
            with open(name, encoding="utf-8") as f:
                texts[name] = f.read()
        except (OSError, UnicodeDecodeError) as e:
            # a file skipped here could be the one holding the broken link, and
            # the run would then close the report issue
            raise ScanError(f"cannot read {name}: {e}") from e
    notebooks, spaces = find_links(texts)
    broken = []
    known: Dict[str, Set[str]] = {}
    for ref, path in sorted(notebooks):
        if ref not in known:  # one request per ref, however many links use it
            known[ref] = notebook_paths(ref)
        if path not in known[ref]:
            where = ", ".join(sorted(notebooks[(ref, path)]))
            at = "" if ref == "master" else f" at {ref}"
            broken.append(f"notebook gone{at}: {path}  (linked from {where})")
    for space_id in sorted(spaces):
        stage = space_stage(space_id)
        if stage in BROKEN_STAGES:
            where = ", ".join(sorted(spaces[space_id]))
            broken.append(f"space {stage}: {space_id}  (linked from {where})")
    print(
        f"checked {len(notebooks)} notebook links and {len(spaces)} Space links",
        file=sys.stderr,
    )
    return broken


def self_check() -> None:
    text = (
        "[a](https://colab.research.google.com/github/espnet/notebook/blob/master/"
        "ESPnet2/Demo/TTS/tts_realtime_demo.ipynb) "
        "[b](https://github.com/espnet/notebook/blob/master/x/y.ipynb) "
        "[c](https://huggingface.co/spaces/espnet/TTS) "
        "[d](https://huggingface.co/spaces) "
        "[e](https://huggingface.co/docs/hub/spaces) "
        "[f](https://colab.research.google.com/assets/colab-badge.svg) "
        "[g](https://huggingface.co/spaces/gradio/omni-mini/tree/main) "
        "[h](https://github.com/espnet/notebook/blob/v1.0/tagged.ipynb)"
    )
    notebooks, spaces = find_links({"README.md": text})
    assert notebooks == {
        ("master", "ESPnet2/Demo/TTS/tts_realtime_demo.ipynb"): {"README.md"},
        ("master", "x/y.ipynb"): {"README.md"},
        ("v1.0", "tagged.ipynb"): {"README.md"},
    }, notebooks
    # the bare /spaces listing, the docs page, the badge image and a link into
    # a Space's files are not links to a running Space
    assert spaces == {"espnet/TTS": {"README.md"}}, spaces
    both = find_links({"a.md": text, "b.md": text})[1]["espnet/TTS"]
    assert both == {"a.md", "b.md"}, both
    print("self-check ok")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--self-check", action="store_true", help="offline scanner test")
    a = p.parse_args()
    if a.self_check:
        self_check()
        return 0
    try:
        broken = scan()
    except ScanError as e:
        print(f"scan failed: {e}", file=sys.stderr)
        return 2
    for line in broken:
        print(line)
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())

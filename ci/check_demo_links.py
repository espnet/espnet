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
import http.client
import json
import os
import re
import subprocess
import sys
import traceback
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

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
# visit, so only these stages are reported. NO_APP_FILE is in the list because
# it is the state an upload that sent the wrong directory leaves behind: the
# repository exists and the page loads, with nothing to run.
BROKEN_STAGES = {
    "RUNTIME_ERROR",
    "BUILD_ERROR",
    "CONFIG_ERROR",
    "NO_APP_FILE",
    "DELETED",
}
TIMEOUT = 30


class ScanError(RuntimeError):
    """The scan could not be completed, which is not the same as a broken link."""


def _git(*args: str, cwd: Optional[str] = None) -> str:
    try:
        out = subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True, cwd=cwd
        )
    except (OSError, subprocess.CalledProcessError) as e:
        # not "nothing is broken": the scan never looked
        raise ScanError(f"git {' '.join(args)} failed: {e}") from e
    return out.stdout


def documentation_files() -> Tuple[str, List[str]]:
    """The repository root, and the tracked files that point people at a demo.

    Docs and shell scripts both do: utils/synth_wav.sh sent beginners to a
    Colab notebook from its help text, with a link that had rotted like the
    rest. Paths come from the root rather than the working directory, so
    running this from a subdirectory scans the whole repository instead of
    quietly reporting that the few files below it are fine.
    """
    root = _git("rev-parse", "--show-toplevel").strip()
    if not root:
        raise ScanError("not inside a git repository")
    listing = _git("ls-files", "--full-name", "*.md", "*.sh", cwd=root)
    return root, [p for p in listing.split("\n") if p]


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
    request = urllib.request.Request(url)
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token and url.startswith("https://api.github.com/"):
        # unauthenticated GitHub allows 60 requests an hour per address, which
        # a shared runner can exhaust; the workflow's own token lifts that
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise ScanError(f"{url}: HTTP {e.code}") from e
    except (
        urllib.error.URLError,
        TimeoutError,
        json.JSONDecodeError,
        # a truncated response raises IncompleteRead, an HTTPException
        http.client.HTTPException,
    ) as e:
        raise ScanError(f"{url}: {e}") from e


def notebook_paths(ref: str) -> Optional[Set[str]]:
    """The files the notebook repository holds at ``ref``, or None if no such ref.

    A slash in a branch name has to be encoded, or the API reads it as a path.
    """
    quoted = urllib.parse.quote(ref, safe="")
    tree = _get_json(
        f"https://api.github.com/repos/{NOTEBOOK_REPO}/git/trees/{quoted}?recursive=1"
    )
    if tree is None:
        return None
    if "tree" not in tree:
        raise ScanError(f"cannot list {NOTEBOOK_REPO} at {ref}")
    if tree.get("truncated"):
        raise ScanError(f"{NOTEBOOK_REPO} file list at {ref} came back truncated")
    return {t["path"] for t in tree["tree"]}


def split_ref(
    ref: str,
    path: str,
    known: Dict[str, Optional[Set[str]]],
    fetch=None,
):
    """Return (ref, path, files) for a link, moving the ref/path boundary.

    A URL says ``blob/<ref>/<path>`` with nothing marking where one ends, so a
    branch with a slash in it - ``blob/feature/demo/x.ipynb`` - parses as the
    ref ``feature``. The split that finds the file wins, so a ref which merely
    exists cannot make a live notebook look gone; failing that, the first ref
    that resolves is used, and the file is reported missing there. ``master``
    costs one request and is cached across links.
    """
    fetch = fetch or notebook_paths
    parts = path.split("/")
    first = None
    for i in range(len(parts)):
        candidate = "/".join([ref] + parts[:i])
        if candidate not in known:
            known[candidate] = fetch(candidate)
        files = known[candidate]
        if files is None:
            continue
        rest = "/".join(parts[i:])
        if rest in files:  # this is the split the link meant
            return candidate, rest, files
        if first is None:  # a ref that exists but does not hold the file
            first = (candidate, rest, files)
    if first is not None:
        return first
    raise ScanError(f"no such ref in {NOTEBOOK_REPO}: {ref} (from {ref}/{path})")


def space_stage(space_id: str) -> str:
    d = _get_json(f"https://huggingface.co/api/spaces/{space_id}")
    if d is None:
        return "DELETED"
    runtime = d.get("runtime")
    if not isinstance(runtime, dict) or not runtime.get("stage"):
        # the Space exists and its state could not be read; saying so beats
        # inventing a stage that is not in BROKEN_STAGES and passing it
        raise ScanError(f"{space_id}: the Hub's answer carries no runtime stage")
    return runtime["stage"]


def scan() -> List[str]:
    root, names = documentation_files()
    texts = {}
    for name in names:
        full = os.path.join(root, name)
        if os.path.islink(full):
            try:
                os.stat(full)  # follows the link
            except OSError as e:
                # three tracked scripts under egs2 are symlinks whose target is
                # not in the repository; they hold no text to scan here. Any
                # other error is a file this scan failed to read, not an empty
                # one - os.path.exists() would have called both of them False.
                if e.errno not in (errno.ENOENT, errno.ENOTDIR):
                    raise ScanError(f"cannot read {name}: {e}") from e
                continue
        try:
            with open(full, encoding="utf-8") as f:
                texts[name] = f.read()
        except (OSError, UnicodeDecodeError) as e:
            # a file skipped here could be the one holding the broken link, and
            # the run would then close the report issue
            raise ScanError(f"cannot read {name}: {e}") from e
    notebooks, spaces = find_links(texts)
    broken = []
    known: Dict[str, Optional[Set[str]]] = {}  # one request per ref, not per link
    for ref, path in sorted(notebooks):
        real_ref, real_path, files = split_ref(ref, path, known)
        if real_path not in files:
            where = ", ".join(sorted(notebooks[(ref, path)]))
            at = "" if real_ref == "master" else f" at {real_ref}"
            broken.append(f"notebook gone{at}: {real_path}  (linked from {where})")
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

    # the ref/path boundary, against a repository where only "feature/demo"
    # exists: the first candidate 404s and the second resolves
    seen = []

    def fake_paths(ref):
        seen.append(ref)
        return {"x.ipynb"} if ref == "feature/demo" else None

    got = split_ref("feature", "demo/x.ipynb", {}, fetch=fake_paths)
    assert got == ("feature/demo", "x.ipynb", {"x.ipynb"}), got
    assert seen == ["feature", "feature/demo"], seen

    # a ref that exists but does not hold the file must not hide a split that
    # does: "master" resolves first, and the file lives on "master/ESPnet2"
    trees = {"master": {"other.ipynb"}, "master/ESPnet2": {"x.ipynb"}}
    got = split_ref("master", "ESPnet2/x.ipynb", {}, fetch=trees.get)
    assert got == ("master/ESPnet2", "x.ipynb", {"x.ipynb"}), got

    # when no split holds the file, it is reported against the ref that exists
    got = split_ref("master", "gone.ipynb", {}, fetch={"master": {"a.ipynb"}}.get)
    assert got == ("master", "gone.ipynb", {"a.ipynb"}), got

    try:
        split_ref("nope", "x.ipynb", {}, fetch=lambda ref: None)
    except ScanError as e:
        assert "no such ref" in str(e), e
    else:  # pragma: no cover - the raise above is the expected path
        raise AssertionError("a ref that does not exist must fail the scan")
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
    except Exception:
        # Anything unforeseen is still "the scan did not run": exiting 1 here
        # would tell the workflow that the links are broken, and it would file
        # a report naming none of them.
        traceback.print_exc()
        print("scan failed: unexpected error", file=sys.stderr)
        return 2
    for line in broken:
        print(line)
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())

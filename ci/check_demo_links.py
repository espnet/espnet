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
import datetime
import email.utils
import errno
import http.client
import io
import json
import os
import re
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import yaml

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
    r"/(?P<name>[A-Za-z0-9_.-]+)/?(?![A-Za-z0-9_.-]|/\S)"
)
# Listed the other way round on purpose. A link is good when clicking it gets
# you the app: running, still starting, or asleep - a visit wakes a sleeping
# Space. Everything else is reported: erroring, being deleted, stopped, paused
# (only its owner can restart one), an upload that sent the wrong directory
# (NO_APP_FILE), and any stage the Hub adds after this was written, which is
# the point - an unrecognised state must not read as a working demo.
WORKING_STAGES = {
    "RUNNING",
    "RUNNING_BUILDING",
    "RUNNING_APP_STARTING",
    "BUILDING",
    "APP_STARTING",
    "SLEEPING",
}
TIMEOUT = 30
# The Hub rate-limits by address - 500 requests in a five-minute window - and
# a shared runner can reach that without this repository having asked for
# anything. A 429 is then a wait rather than a broken link, and the response
# says how long to wait, so it is waited out once instead of failing the scan.
RATE_LIMITED = {429, 503}
# The Hub also answers an anonymous request with 401 under load. Nothing here
# sends credentials, so a 401 cannot be about credentials, and it is not about
# the link either: on 2026-09-23 one took the daily run red on
# api/spaces/espnet/forced-alignment, a Space that is public, ungated and
# RUNNING, and the same scan passed the next morning. One of seven runs.
DECLINED = {401}
# What is worth one retry. Kept separate from RATE_LIMITED because only that
# one is throttling, and only that one names a wait to honour.
TRANSIENT = RATE_LIMITED | DECLINED
# One window, and no more. Waiting five minutes in a job that otherwise takes
# seconds is still cheaper than a red daily run that someone has to open to
# find out it was throttling; a wait longer than a window is not this
# repository being throttled and is reported instead.
MAX_WAIT = 300
# what to wait when the server throttles without saying for how long, which is
# what a 503 from either API usually looks like
DEFAULT_WAIT = 30


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
            # a URL may carry %2F and friends; notebook_paths re-encodes the
            # ref itself, and the tree lists decoded paths
            key = (
                urllib.parse.unquote(m.group("ref")),
                urllib.parse.unquote(m.group("path")),
            )
            notebooks[key].add(name)
        for m in SPACE_LINK.finditer(text):
            spaces[f"{m.group('owner')}/{m.group('name')}"].add(name)
    return dict(notebooks), dict(spaces)


def retry_delay(headers) -> Optional[float]:
    """Seconds the server asked us to wait, or None if it did not say.

    Three spellings: `Retry-After: 30`, the same header as an HTTP date, which
    RFC 9110 allows and which a 30 second guess would not be long enough for,
    and the RateLimit header's `t` field, which is what the Hub sends -
    `ratelimit: "api";r=0;t=231` is 231 seconds until the window reopens.
    """
    after = headers.get("Retry-After") if headers else None
    if after:
        try:
            return max(0.0, float(after.strip()))
        except ValueError:
            pass
        try:
            when = email.utils.parsedate_to_datetime(after)
        except (TypeError, ValueError):
            when = None
        if when is not None:
            # a date without a zone is read as UTC, which is what the header
            # is required to carry anyway
            if when.tzinfo is None:
                when = when.replace(tzinfo=datetime.timezone.utc)
            now = datetime.datetime.now(datetime.timezone.utc)
            return max(0.0, (when - now).total_seconds())
    limit = headers.get("RateLimit") if headers else None
    if limit:
        m = re.search(r"\bt\s*=\s*(\d+(?:\.\d+)?)", limit)
        if m:
            return float(m.group(1))
    return None


def _get_json(url: str, sleep=time.sleep):
    request = urllib.request.Request(url)
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    authenticated = bool(token) and url.startswith("https://api.github.com/")
    if authenticated:
        # unauthenticated GitHub allows 60 requests an hour per address, which
        # a shared runner can exhaust; the workflow's own token lifts that
        request.add_header("Authorization", f"Bearer {token}")
    for attempt in (1, 2):
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            wait = retry_delay(e.headers) if e.code in RATE_LIMITED else None
            # A 401 is only worth retrying where no credentials were sent.
            # Retrying a rejected token sends the same rejected token again.
            retryable = e.code in RATE_LIMITED or (
                e.code in DECLINED and not authenticated
            )
            if attempt == 1 and retryable:
                # a server that throttles without saying for how long still
                # deserves the one retry; only a wait it named and that is
                # longer than a window skips it
                naps = DEFAULT_WAIT if wait is None else wait
                if naps <= MAX_WAIT:
                    sleep(naps)
                    continue
            if e.code in RATE_LIMITED:
                # said plainly: a red job here is the address being throttled,
                # and no link in this repository has been shown to be wrong
                said = f" and asked for {wait:.0f}s" if wait is not None else ""
                raise ScanError(
                    f"{url}: HTTP {e.code} - rate limited{said}, not a broken link"
                ) from e
            if e.code in DECLINED:
                if authenticated:
                    # this one *did* send credentials, so it is an
                    # authentication failure and saying "not a broken link"
                    # would hide the thing worth acting on
                    raise ScanError(
                        f"{url}: HTTP {e.code} - the token this request sent "
                        "was rejected; check GH_TOKEN"
                    ) from e
                # not "rate limited", which it does not say it is, and not a
                # broken link either - this request carried no credentials to
                # be wrong about
                raise ScanError(
                    f"{url}: HTTP {e.code} - the request was declined, "
                    "not a broken link"
                ) from e
            raise ScanError(f"{url}: HTTP {e.code}") from e
        except (
            urllib.error.URLError,
            TimeoutError,
            json.JSONDecodeError,
            # a truncated response raises IncompleteRead, an HTTPException
            http.client.HTTPException,
        ) as e:
            raise ScanError(f"{url}: {e}") from e
    raise AssertionError("unreachable: the loop returns or raises")


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


# A Space this repository holds the source of, but which has not been
# uploaded yet. Its card names the release it needs, a Space installs espnet
# from PyPI, and the cards say plainly that an upload before that release
# replaces a working demo with a broken one - so until the release is out,
# the Space being absent is the rule working, not a broken link.
ABSENT = "NOT PUBLISHED"


def space_stage(space_id: str) -> str:
    try:
        d = _get_json(f"https://huggingface.co/api/spaces/{space_id}")
    except ScanError as e:
        # the Hub answers 401 for a Space that does not exist, so that a
        # private one cannot be told from a missing one
        if " HTTP 401" in str(e):
            return ABSENT
        raise
    if d is None:
        return "DELETED"
    runtime = d.get("runtime")
    if not isinstance(runtime, dict) or not runtime.get("stage"):
        # the Space exists and its state could not be read; saying so beats
        # inventing a stage that is not in BROKEN_STAGES and passing it
        raise ScanError(f"{space_id}: the Hub's answer carries no runtime stage")
    return runtime["stage"]


# What the Hub refuses in a Space's README front matter. Every one of these
# stops `hf upload` before a single file lands, and the demo card in this
# repository is written here rather than on the Hub, so nothing else checks it.
SPACE_CARD_LIMITS = {"short_description": 60, "title": 100}
SPACE_CARD_REQUIRED = ("title", "sdk", "app_file")


def awaiting_release(path: str, text: str, root: str = ".") -> Optional[str]:
    """The espnet release a Space card is waiting for, if it is not out yet.

    A card names the Space it will be uploaded to before that Space exists -
    that is the order the cards themselves prescribe. This says which release
    the directory pins and whether PyPI has it, so an absent Space can be
    reported as pending rather than as a broken link.
    """
    pinned = None
    requirements = os.path.join(root, os.path.dirname(path), "requirements.txt")
    try:
        with open(requirements, encoding="utf-8") as f:
            for line in f:
                # one version, not the rest of the requirement:
                # `espnet>=a,!=b` would otherwise ask PyPI for the release
                # "a,!=b", get nothing, and call a published Space pending
                match = re.match(
                    r"\s*espnet(\[[a-z,]+\])?(>=|==)" r"(?P<version>[A-Za-z0-9._+!-]+)",
                    line,
                )
                if match:
                    pinned = match.group("version")
                    break
    except OSError:
        return None
    if pinned is None:
        return None
    released = _get_json(f"https://pypi.org/pypi/espnet/{pinned}/json")
    return None if released else pinned


# The line every card carries, and the only statement in it of where this
# directory goes. A card also links to its sibling - the two OWSM demos point
# at each other - so the links cannot say which Space is the card's own, and
# taking them all would have hidden a real outage of the sibling behind
# "not published yet".
UPLOAD = re.compile(
    r"^hf upload (?P<space>[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+) (?P<dir>\S+) ",
    re.M,
)


def upload_target(path: str, text: str) -> Optional[str]:
    """The Space this card says its own directory is uploaded to.

    None when the card does not say, or says it of another directory. Then
    nothing about it is treated as pending, which is the safe direction: an
    absent Space is reported rather than excused.
    """
    match = UPLOAD.search(text)
    if match is None:
        return None
    if match.group("dir").rstrip("/") != os.path.dirname(path).rstrip("/"):
        return None
    return match.group("space")


# `demo/` is one Space in a recipe, `demo_align/` a second one beside it -
# a recipe whose model serves two demos of different shapes. Matching only
# `/demo/` left the second card unchecked: neither its front matter, which
# the Hub refuses silently, nor the Space it names.
DEMO_DIR = re.compile(r"/demo(_[a-z0-9]+)?/")


def space_cards(names: List[str]) -> List[str]:
    """Paths of the Space READMEs among the tracked files."""
    return [p for p in names if p.endswith("README.md") and DEMO_DIR.search(f"/{p}")]


def check_space_card(path: str, text: str, root: str = ".") -> List[str]:
    """Report what the Hub would refuse in this Space card's front matter.

    ``path`` is relative to ``root``, and so is everything resolved from it:
    the check must say the same thing wherever it is run from.
    """
    if not text.startswith("---"):
        return []  # not a Space card, just a README in a demo directory
    parts = text.split("---", 2)
    if len(parts) < 3:
        # the Hub reads metadata only between two delimiters; without the
        # closing one it sees none at all, whatever the text says
        return [f"{path}: front matter is opened with --- but never closed"]
    try:
        front = yaml.safe_load(parts[1])
    except yaml.YAMLError as e:
        return [f"{path}: front matter is not valid YAML: {e}"]
    if not isinstance(front, dict) or "sdk" not in front:
        return []

    problems = []
    for key in SPACE_CARD_REQUIRED:
        value = front.get(key)
        # YAML turns `title: true` into a bool and `sdk: 1` into an int, and
        # the Hub wants text in both
        if not isinstance(value, str) or not value.strip():
            problems.append(f"{path}: the Hub requires {key} in a Space card, as text")
    for key, limit in SPACE_CARD_LIMITS.items():
        value = front.get(key)
        if isinstance(value, str) and len(value) > limit:
            problems.append(
                f"{path}: {key} is {len(value)} characters; the Hub allows {limit}"
            )

    app = front.get("app_file")
    if isinstance(app, str) and app.strip():
        # the directory uploaded as the Space is the Space's root, so app_file
        # has to stay inside it
        directory = (Path(root) / path).parent.resolve()
        target = (directory / app).resolve()
        # is_relative_to, not a string prefix: a sibling directory whose name
        # starts the same way - demo-copy beside demo - passes a prefix test
        if not target.is_relative_to(directory):
            problems.append(f"{path}: app_file {app} points outside the Space")
        elif not target.exists():
            problems.append(f"{path}: app_file {app} is not next to it")
    return problems


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
    for path in space_cards(names):
        # an upload is refused outright for these, so the demo never appears
        broken.extend(check_space_card(path, texts.get(path, ""), root))
    known: Dict[str, Optional[Set[str]]] = {}  # one request per ref, not per link
    for ref, path in sorted(notebooks):
        real_ref, real_path, files = split_ref(ref, path, known)
        if real_path not in files:
            where = ", ".join(sorted(notebooks[(ref, path)]))
            at = "" if real_ref == "master" else f" at {real_ref}"
            broken.append(f"notebook gone{at}: {real_path}  (linked from {where})")
    # a Space whose source is here and whose card pins a release PyPI does
    # not have cannot have been uploaded yet
    pending = {}
    for path in space_cards(names):
        target = upload_target(path, texts.get(path, ""))
        waiting = awaiting_release(path, texts.get(path, ""), root) if target else None
        if waiting:
            pending[target] = (waiting, path)
    for space_id in sorted(spaces):
        stage = space_stage(space_id)
        if stage == ABSENT and space_id in pending:
            waiting, card = pending[space_id]
            print(
                f"not published yet: {space_id} waits for espnet {waiting} "
                f"({card} says so)",
                file=sys.stderr,
            )
            continue
        if stage not in WORKING_STAGES:
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
        "Demos/TTS/tts_realtime_demo.ipynb) "
        "[b](https://github.com/espnet/notebook/blob/master/x/y.ipynb) "
        "[c](https://huggingface.co/spaces/espnet/TTS) "
        "[d](https://huggingface.co/spaces) "
        "[e](https://huggingface.co/docs/hub/spaces) "
        "[f](https://colab.research.google.com/assets/colab-badge.svg) "
        "[g](https://huggingface.co/spaces/gradio/omni-mini/tree/main) "
        "[h](https://github.com/espnet/notebook/blob/v1.0/tagged.ipynb) "
        "[i](https://huggingface.co/spaces/espnet/svs/) "
        "[j](https://github.com/espnet/notebook/blob/feature%2Fdemo/a%20b.ipynb)"
    )
    notebooks, spaces = find_links({"README.md": text})
    assert notebooks == {
        ("master", "Demos/TTS/tts_realtime_demo.ipynb"): {"README.md"},
        ("master", "x/y.ipynb"): {"README.md"},
        ("v1.0", "tagged.ipynb"): {"README.md"},
        # percent-encoding is undone here, since the ref is encoded again when
        # its tree is fetched and the tree's paths come back decoded
        ("feature/demo", "a b.ipynb"): {"README.md"},
    }, notebooks
    # the bare /spaces listing, the docs page, the badge image and a link into
    # a Space's files are not links to a running Space
    assert spaces == {
        "espnet/TTS": {"README.md"},
        # a root URL with a trailing slash is still a link to the app
        "espnet/svs": {"README.md"},
    }, spaces
    both = find_links({"a.md": text, "b.md": text})[1]["espnet/TTS"]
    assert both == {"a.md", "b.md"}, both

    # a Space card the Hub would refuse: 67 characters where it allows 60,
    # which is what stopped the OWSM-CTC demo's first upload
    card = (
        "---\ntitle: OWSM-CTC v4\nsdk: gradio\napp_file: app.py\n"
        "short_description: Multilingual ASR, speech translation and language ID"
        " in one encoder\n---\n"
    )
    problems = check_space_card("egs2/x/demo/README.md", card)
    assert any("short_description is 67 characters" in p for p in problems), problems
    assert any("app_file app.py is not next to it" in p for p in problems), problems
    ok = card.replace(
        "Multilingual ASR, speech translation and language ID in one encoder",
        "ASR, speech translation and language ID in one encoder",
    ).replace("app_file: app.py\n", "")
    assert check_space_card("egs2/x/demo/README.md", ok) == [
        "egs2/x/demo/README.md: the Hub requires app_file in a Space card, as text"
    ], check_space_card("egs2/x/demo/README.md", ok)
    # a plain README in a demo directory is not a Space card
    assert check_space_card("egs2/x/demo/README.md", "# just a readme\n") == []

    # YAML types: `title: true` is a bool and `sdk: 1` an int, and the Hub
    # wants text - truthiness alone would have let both through
    typed = "---\ntitle: true\nsdk: 1\napp_file: app.py\n---\n"
    problems = check_space_card("egs2/x/demo/README.md", typed)
    assert any("requires title" in p for p in problems), problems
    assert any("requires sdk" in p for p in problems), problems

    # front matter that is never closed carries no metadata to the Hub
    unclosed = "---\ntitle: x\nsdk: gradio\napp_file: app.py\n"
    assert check_space_card("egs2/x/demo/README.md", unclosed) == [
        "egs2/x/demo/README.md: front matter is opened with --- but never closed"
    ], check_space_card("egs2/x/demo/README.md", unclosed)

    # app_file has to stay inside the directory that becomes the Space
    escaping = "---\ntitle: x\nsdk: gradio\napp_file: ../outside.py\n---\n"
    assert check_space_card("egs2/x/demo/README.md", escaping) == [
        "egs2/x/demo/README.md: app_file ../outside.py points outside the Space"
    ], check_space_card("egs2/x/demo/README.md", escaping)

    # a sibling whose name begins with this directory's name is outside it too
    sibling = "---\ntitle: x\nsdk: gradio\napp_file: ../demo-copy/app.py\n---\n"
    assert check_space_card("egs2/x/demo/README.md", sibling) == [
        "egs2/x/demo/README.md: app_file ../demo-copy/app.py points outside the Space"
    ], check_space_card("egs2/x/demo/README.md", sibling)

    # a non-string app_file is reported, not a TypeError that ends the scan
    weird = "---\ntitle: x\nsdk: gradio\napp_file: true\n---\n"
    assert check_space_card("egs2/x/demo/README.md", weird) == [
        "egs2/x/demo/README.md: the Hub requires app_file in a Space card, as text"
    ], check_space_card("egs2/x/demo/README.md", weird)

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

    # a rate limit is waited out once, with the wait the server named, and
    # only then reported - the daily run went red on a 429 from a shared
    # runner while every link in the repository was fine
    assert retry_delay(email.message_from_string("Retry-After: 30")) == 30
    assert retry_delay(email.message_from_string('RateLimit: "api";r=0;t=231')) == 231
    assert retry_delay(email.message_from_string("")) is None
    assert retry_delay(email.message_from_string("Retry-After: Wed, 21 Oct")) is None

    # the date spelling RFC 9110 allows. Guessing 30 seconds at one of these
    # would retry while still throttled, which is the failure this exists to
    # avoid, so it is read rather than ignored
    soon = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(
        seconds=120
    )
    dated = email.message_from_string(
        f"Retry-After: {email.utils.format_datetime(soon)}"
    )
    assert 110 <= retry_delay(dated) <= 120, retry_delay(dated)
    past = email.message_from_string("Retry-After: Wed, 21 Oct 2015 07:28:00 GMT")
    assert retry_delay(past) == 0.0, retry_delay(past)

    calls = []
    slept = []

    def rate_limited(url, **_):
        calls.append(url)
        if len(calls) == 1:
            raise urllib.error.HTTPError(
                url,
                429,
                "Too Many Requests",
                email.message_from_string('RateLimit: "api";r=0;t=7'),
                None,
            )
        return io.BytesIO(b'{"ok": true}')

    real_urlopen = urllib.request.urlopen
    urllib.request.urlopen = rate_limited
    try:
        got = _get_json("https://huggingface.co/api/spaces/x/y", sleep=slept.append)
    finally:
        urllib.request.urlopen = real_urlopen
    assert got == {"ok": True}, got
    assert slept == [7.0], slept
    assert len(calls) == 2, calls

    def always_limited(url, **_):
        raise urllib.error.HTTPError(
            url,
            429,
            "Too Many Requests",
            email.message_from_string('RateLimit: "api";r=0;t=9'),
            None,
        )

    urllib.request.urlopen = always_limited
    try:
        _get_json("https://huggingface.co/api/spaces/x/y", sleep=lambda _: None)
    except ScanError as e:
        assert "rate limited" in str(e) and "not a broken link" in str(e), e
    else:  # pragma: no cover - the raise above is the expected path
        raise AssertionError("a rate limit that does not clear must fail the scan")
    finally:
        urllib.request.urlopen = real_urlopen

    # a 401 gets the same one retry. The Hub answers anonymous requests with
    # one under load, and on 2026-09-23 that alone took the daily run red
    # while every link in the repository was fine.
    unauthorised = []

    def declined_once(url, **_):
        unauthorised.append(url)
        if len(unauthorised) == 1:
            raise urllib.error.HTTPError(
                url, 401, "Unauthorized", email.message_from_string(""), None
            )
        return io.BytesIO(b'{"ok": true}')

    real_urlopen = urllib.request.urlopen
    urllib.request.urlopen = declined_once
    napped = []
    try:
        got = _get_json("https://huggingface.co/api/spaces/x/y", sleep=napped.append)
    finally:
        urllib.request.urlopen = real_urlopen
    assert got == {"ok": True}, got
    assert napped == [DEFAULT_WAIT], napped
    assert len(unauthorised) == 2, unauthorised

    # and one that does not clear is still the scan failing, not a broken
    # link - the distinction the exit status exists for
    def always_declined(url, **_):
        raise urllib.error.HTTPError(
            url, 401, "Unauthorized", email.message_from_string(""), None
        )

    urllib.request.urlopen = always_declined
    try:
        _get_json("https://huggingface.co/api/spaces/x/y", sleep=lambda _: None)
    except ScanError as e:
        assert "declined" in str(e) and "not a broken link" in str(e), e
        assert "rate limited" not in str(e), e
    else:  # pragma: no cover - the raise above is the expected path
        raise AssertionError("a 401 that does not clear must fail the scan")
    finally:
        urllib.request.urlopen = real_urlopen

    # a 401 on a request that *did* send a token is an authentication
    # failure, not the Hub declining an anonymous caller: retrying sends the
    # same rejected token again, and "not a broken link" would hide the thing
    # worth acting on
    authed = []

    def rejects_token(request, **_):
        # the Authorization header itself, not just the fact that a token was
        # in the environment: without this the case passes with the
        # add_header call deleted, which is the one thing it is here to check
        authed.append(request.get_header("Authorization"))
        raise urllib.error.HTTPError(
            request.full_url, 401, "Unauthorized", email.message_from_string(""), None
        )

    real_urlopen = urllib.request.urlopen
    urllib.request.urlopen = rejects_token
    had = os.environ.get("GH_TOKEN")
    os.environ["GH_TOKEN"] = "not-a-real-token"
    slept_on_auth = []
    try:
        _get_json(
            "https://api.github.com/repos/espnet/notebook/git/trees/master",
            sleep=slept_on_auth.append,
        )
    except ScanError as e:
        assert "token" in str(e) and "GH_TOKEN" in str(e), e
        assert "not a broken link" not in str(e), e
    else:  # pragma: no cover - the raise above is the expected path
        raise AssertionError("a rejected token must fail the scan")
    finally:
        urllib.request.urlopen = real_urlopen
        if had is None:
            del os.environ["GH_TOKEN"]
        else:
            os.environ["GH_TOKEN"] = had
    assert len(authed) == 1, authed
    assert authed[0] == "Bearer not-a-real-token", authed
    assert slept_on_auth == [], slept_on_auth

    # a 403 is not in either set: it gets no retry and no reassurance
    forbidden = []

    def always_forbidden(url, **_):
        forbidden.append(url)
        raise urllib.error.HTTPError(
            url, 403, "Forbidden", email.message_from_string(""), None
        )

    urllib.request.urlopen = always_forbidden
    try:
        _get_json("https://huggingface.co/api/spaces/x/y", sleep=lambda _: None)
    except ScanError as e:
        assert "403" in str(e) and "not a broken link" not in str(e), e
    else:  # pragma: no cover - the raise above is the expected path
        raise AssertionError("a 403 must fail the scan")
    finally:
        urllib.request.urlopen = real_urlopen
    assert len(forbidden) == 1, forbidden

    # a 503 that says nothing still gets the one retry, after the default
    third = []

    def silent_503(url, **_):
        third.append(url)
        if len(third) == 1:
            raise urllib.error.HTTPError(
                url, 503, "Service Unavailable", email.message_from_string(""), None
            )
        return io.BytesIO(b'{"ok": true}')

    urllib.request.urlopen = silent_503
    waited = []
    try:
        got = _get_json("https://huggingface.co/api/spaces/x/y", sleep=waited.append)
    finally:
        urllib.request.urlopen = real_urlopen
    assert got == {"ok": True} and waited == [DEFAULT_WAIT], (got, waited)

    # a wait longer than a CI minute is not waited out at all
    def far_off(url, **_):
        raise urllib.error.HTTPError(
            url,
            429,
            "Too Many Requests",
            email.message_from_string(f"Retry-After: {MAX_WAIT + 1}"),
            None,
        )

    urllib.request.urlopen = far_off
    try:
        _get_json("https://huggingface.co/api/spaces/x/y", sleep=lambda _: 1 / 0)
    except ScanError as e:
        assert f"asked for {MAX_WAIT + 1}s" in str(e), e
    else:  # pragma: no cover - the raise above is the expected path
        raise AssertionError("a long wait must be reported, not slept through")
    finally:
        urllib.request.urlopen = real_urlopen

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

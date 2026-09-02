#!/usr/bin/env python3

"""Prepare an ESPnet release.

Checks it is safe, writes the notes, and stops before the point of no return.

Replaces doc/make_release_note_from_milestone.py, which only generated notes.

    pip install PyGithub
    # check, and print the notes
    python doc/make_release.py <github_token> <milestone>

    # also write version.txt and create or update the draft release
    python doc/make_release.py <github_token> <milestone> --apply

Everything this does is reversible. It never creates a tag and never uploads
anything, because pushing a `v*` tag triggers publish_python_package.yml, which
uploads to PyPI, and PyPI will not accept the same version twice. The last step
is left to a person, and the script prints it.

The checks exist because the 202604 release failed quietly in three different
ways at once, and nothing said so:

  - The milestone was named v.202607 while the release was going out as 202609,
    so "everything in the milestone" and "everything since the last tag" were
    two different sets of pull requests, overlapping by a third.
  - 21 pull requests were still open in the milestone.
  - The PyPI upload had failed on both v.202604 and v.202604-patch1, so PyPI's
    newest espnet was five months older than GitHub's. Nobody found out until
    somebody went looking.

So: check first, and fail with the reason rather than producing a release note
for a release that cannot happen.
"""

import argparse
import json
import re
import subprocess
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

import github

REPO_ROOT = Path(__file__).resolve().parent.parent
VERSION_FILE = REPO_ROOT / "version.txt"
PUBLISH_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "publish_python_package.yml"

# The labels the notes are grouped under, in the order they appear.
PICKUP_LABELS = [
    "New Features",
    "Enhancement",
    "Recipe",
    "Bugfix",
    "Documentation",
    "Refactoring",
]


def sh(*args):
    """Run a command in the repository and return its stdout, or None if it failed."""
    result = subprocess.run(
        args, cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def origin_slug():
    """owner/repo for the checkout's origin remote."""
    url = sh("git", "remote", "get-url", "origin")
    if not url:
        sys.exit("no `origin` remote; run this from an espnet checkout")
    match = re.search(r"[:/]([^/:]+)/([^/]+?)(?:\.git)?$", url)
    if not match:
        sys.exit(f"cannot read owner/repo out of the origin remote: {url}")
    return f"{match.group(1)}/{match.group(2)}"


def package_name():
    """The distribution name, so the PyPI check follows the project being released."""
    pyproject = (REPO_ROOT / "pyproject.toml").read_text()
    match = re.search(r'^name\s*=\s*"([^"]+)"', pyproject, re.M)
    return match.group(1) if match else "espnet"


def version_of(milestone_title):
    """v.202609 -> 202609. version.txt holds the number without the prefix."""
    return milestone_title.removeprefix("v.").removeprefix("v")


def pypi_has(version):
    """True if PyPI already serves this version, False if not, None if unknown.

    None matters: a network failure must not read as "the version is free".
    """
    url = f"https://pypi.org/pypi/{package_name()}/json"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return version in json.load(response)["releases"]
    except (urllib.error.URLError, json.JSONDecodeError, KeyError, TimeoutError):
        return None


def trusted_publishing_ready():
    """Report why publish_python_package.yml would not publish, if it would not.

    Checked because the credential-based version of this workflow failed twice in
    a row without anyone noticing. These are the two properties trusted
    publishing needs, and both are invisible at release time.
    """
    if not PUBLISH_WORKFLOW.is_file():
        return [f"{PUBLISH_WORKFLOW.name} is missing"]
    text = PUBLISH_WORKFLOW.read_text()
    problems = []
    if "id-token: write" not in text:
        problems.append(
            f"{PUBLISH_WORKFLOW.name} has no `id-token: write`, which PyPI's "
            "trusted publishing requires"
        )
    if "secrets." in text:
        problems.append(
            f"{PUBLISH_WORKFLOW.name} still reads a secret; passing a username or "
            "password takes the upload out of the trusted publishing flow"
        )
    return problems


def preflight(repo, milestone, version, open_items, will_apply):
    """Everything that has to be true before a release can go out.

    will_apply excuses the one problem --apply exists to fix. Without it the
    documented `--apply` run was unreachable: the version.txt mismatch is exactly
    the state you start from, it failed preflight, and preflight exits before
    apply_changes runs - so the only way through was --force, which also waves
    past the checks that matter.
    """
    problems = []

    declared = VERSION_FILE.read_text().strip() if VERSION_FILE.is_file() else None
    if declared != version and not will_apply:
        problems.append(
            f"version.txt says {declared!r}, milestone {milestone.title} implies "
            f"{version!r} - one of the two is wrong (--apply writes version.txt)"
        )

    if open_items:
        listed = ", ".join(f"#{i.number}" for i in open_items[:5])
        more = f" and {len(open_items) - 5} more" if len(open_items) > 5 else ""
        problems.append(
            f"{len(open_items)} open item(s) in {milestone.title}: {listed}{more} - "
            "merge them or move them to the next milestone"
        )

    tag = f"v.{version}"
    if any(t.name == tag for t in repo.get_tags()):
        problems.append(f"tag {tag} already exists; this release has been cut")

    on_pypi = pypi_has(version)
    if on_pypi is True:
        problems.append(
            f"PyPI already serves {package_name()} {version}; it will refuse the "
            "upload, and a version cannot be replaced"
        )
    elif on_pypi is None:
        problems.append("could not reach PyPI to check whether this version exists")

    problems += trusted_publishing_ready()

    branch = sh("git", "rev-parse", "--abbrev-ref", "HEAD")
    if branch != "master":
        problems.append(f"on branch {branch!r}, not master")
    if sh("git", "status", "--porcelain"):
        problems.append("the working tree has uncommitted changes")

    return problems


def collect(repo, milestone):
    """Merged pull requests in the milestone, grouped by label, plus contributors."""
    merged = []
    open_items = []
    for issue in repo.get_issues(milestone=milestone, state="all"):
        if issue.state == "open":
            open_items.append(issue)
            continue
        try:
            pull = issue.as_pull_request()
        except github.UnknownObjectException:
            continue
        if pull.merged:
            merged.append(pull)

    grouped = defaultdict(list)
    contributors = []
    for pull in merged:
        if pull.user.login not in contributors:
            contributors.append(pull.user.login)
        # A pull request with no labels, or none of ours, goes to Others. The
        # previous version of this decided that with a flag set inside the label
        # loop, so an unlabelled pull request read the value left over from the
        # one before it.
        names = [label.name for label in pull.labels]
        label = next((n for n in names if n in PICKUP_LABELS), "Others")
        grouped[label].append(pull)
    return grouped, contributors, merged, open_items


def render(milestone, grouped, contributors, merged, previous_tag):
    """The release note. Mechanical parts filled in, the summary left to a person."""
    out = []
    version = version_of(milestone.title)
    out.append("# Summary\n")
    out.append("## Overview\n")
    out.append(
        "<!-- Replace this paragraph. What is the release actually about? The "
        "sections below are mechanical; this is the part that needs a person. "
        "Lead with the requirement changes if there are any - a date-based "
        "version number gives no hint of them. -->\n"
    )

    out.append("## Important PRs\n")
    for label in PICKUP_LABELS + ["Others"]:
        if label not in grouped:
            continue
        out.append(f"### {label}\n")
        for pull in grouped[label]:
            out.append(f"- **PR #{pull.number}**: {pull.title} (by @{pull.user.login})")
        out.append("")

    out.append("---\n")
    out.append("## Contributors\n")
    humans = [c for c in contributors if not c.endswith("[bot]")]
    bots = [c for c in contributors if c.endswith("[bot]")]
    out.append(
        f"{len(humans)} contributors, across {len(merged)} merged pull requests "
        f"in {milestone.title}.\n"
    )
    out.append(", ".join(f"@{c}" for c in sorted(humans, key=str.lower)) + ".")
    if bots:
        out.append("\nPlus " + ", ".join(f"@{b}" for b in sorted(bots)) + ".")
    if previous_tag:
        out.append(
            f"\n**Full changelog**: "
            f"https://github.com/espnet/espnet/compare/{previous_tag}...v.{version}"
        )
    return "\n".join(out) + "\n"


def apply_changes(repo, milestone, version, notes):
    """The reversible half: version.txt, and a draft release holding the notes."""
    if VERSION_FILE.read_text().strip() != version:
        VERSION_FILE.write_text(version + "\n")
        print(f"wrote version.txt = {version}")
        print("  commit it, open a pull request, and merge before tagging")

    title = f"ESPnet version {version}"
    for release in repo.get_releases():
        if release.tag_name == f"v.{version}":
            release.update_release(
                name=title, message=notes, draft=release.draft, prerelease=False
            )
            print(f"updated the existing release for v.{version}: {release.html_url}")
            return
    draft = repo.create_git_release(
        tag=f"v.{version}", name=title, message=notes, draft=True
    )
    print(f"created a draft release: {draft.html_url}")
    print("  a draft creates no tag; publishing it does")


def main():
    parser = argparse.ArgumentParser("prepare an ESPnet release")
    parser.add_argument("token", help="GitHub token with repo access")
    parser.add_argument("milestone", help="milestone title, e.g. v.202609")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="also write version.txt and create or update the draft release",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="carry on despite failed checks; prints them and continues",
    )
    parser.add_argument("--output", help="write the notes here instead of stdout")
    args = parser.parse_args()

    # Derived from the checkout, not from a flag. This script releases the tree
    # it lives in - it reads version.txt, the publish workflow and git status from
    # here - so the GitHub repository has to be the one this tree came from. The
    # old script took --user and --repo and then used --user for both, which was a
    # bug; taking them and honouring them would have been a worse one, because
    # --repo other would have checked espnet's local state and drafted a release
    # somewhere else.
    slug = origin_slug()
    client = github.Github(auth=github.Auth.Token(args.token))
    repo = client.get_repo(slug)

    milestone = next(
        (m for m in repo.get_milestones(state="all") if m.title == args.milestone),
        None,
    )
    if milestone is None:
        titles = ", ".join(m.title for m in repo.get_milestones(state="all"))
        sys.exit(f"no milestone titled {args.milestone!r}. Existing: {titles}")

    version = version_of(milestone.title)
    if not re.fullmatch(r"\d{6}(-patch\d+)?", version):
        sys.exit(
            f"milestone {milestone.title!r} does not look like a release: expected "
            "v.YYYYMM, optionally with -patchN"
        )

    grouped, contributors, merged, open_items = collect(repo, milestone)

    problems = preflight(repo, milestone, version, open_items, args.apply)
    if problems:
        print(
            f"{len(problems)} problem(s) before {milestone.title} can ship:\n",
            file=sys.stderr,
        )
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        print("", file=sys.stderr)
        if not args.force:
            sys.exit("nothing written. Fix these, or pass --force.")
        print("continuing anyway (--force)\n", file=sys.stderr)
    else:
        print(
            f"{milestone.title}: ready ({len(merged)} merged pull requests)\n",
            file=sys.stderr,
        )

    tags = [t.name for t in repo.get_tags()]
    previous_tag = next((t for t in tags if t != f"v.{version}"), None)
    notes = render(milestone, grouped, contributors, merged, previous_tag)

    if args.output:
        Path(args.output).write_text(notes)
        print(f"notes written to {args.output}", file=sys.stderr)
    elif not args.apply:
        print(notes)

    if args.apply:
        apply_changes(repo, milestone, version, notes)

    print(
        "\nRemaining, by hand:\n"
        "  1. Merge the version.txt bump.\n"
        "  2. Fill in the Overview in the draft release.\n"
        f"  3. Publish the draft. That creates tag v.{version}, which triggers\n"
        "     publish_python_package.yml and uploads to PyPI. It cannot be undone.\n"
        "  4. Check the run: gh run list --workflow publish_python_package.yml\n"
        "     Both 202604 tags failed here silently. Do not assume it worked.\n"
        "  5. Move anything still open to the next milestone and close this one.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()

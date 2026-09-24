#!/usr/bin/env python3

"""Prepare an ESPnet release.

Checks it is safe, writes the notes, and stops before the point of no return.

Replaces doc/make_release_note_from_milestone.py, which only generated notes.

    pip install PyGithub
    # check, and print the notes
    python doc/make_release.py <github_token> <milestone>

    # also write version.txt and the README entry, and create or update
    # the draft release
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

A patch is v.YYYYMM.postN, and the tag is the milestone title. The older tags
spell it -patchN, which is not a version PyPI accepts, so their version.txt
had to say something else - and v.202609-patch1 forgot to, leaving the tag
and the published version one apart for the rest of that series.

So: check first, and fail with the reason rather than producing a release note
for a release that cannot happen.
"""

import argparse
import contextlib
import importlib.util
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
README_FILE = REPO_ROOT / "README.md"
PUBLISH_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "publish_python_package.yml"

# What the What's new entry says until a person writes the real summary. A
# release that still carries it has a README that says nothing about itself.
README_PLACEHOLDER = "TODO: what this release is about, in one or two lines."

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


# A release is v.YYYYMM, and a patch on top of one is v.YYYYMM.postN.
#
# .postN and not -patchN, because version.txt is what the package is published
# as and `202610-patch1` is not a version PyPI accepts; PEP 440 spells it
# `202610.post1`. The tags before 202610 say -patchN and their version.txt
# says .postN - and off by one at that, because v.202609-patch1 never bumped
# the file at all. Deriving the version from the milestone title is what stops
# the tag and the published version drifting apart again, and that only works
# if the title is already the version.
# [0-9] and not \d: \d matches every Unicode decimal digit, so a milestone
# typed with fullwidth digits - easy enough on a Japanese keyboard - would
# pass this and then be refused by PyPI, which is the failure this check
# exists to catch early.
RELEASE_VERSION = re.compile(r"[0-9]{6}(\.post[0-9]+)?")


def release_version(milestone_title):
    """The version this milestone releases, or None if it names no release."""
    version = version_of(milestone_title)
    return version if RELEASE_VERSION.fullmatch(version) else None


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


def publishable_metadata():
    """Reject a direct reference in pyproject.toml before the tag goes out.

    PyPI refuses any distribution whose metadata declares one, and nothing local
    says so first: the build succeeds and `twine check` reports PASSED on the
    rejected artifact. This is what silently held espnet off PyPI from v.202604
    to v.202609. The rule lives in the CI checker so it also fails on the pull
    request that introduces it; this reuses it rather than restating it.
    """
    checker = REPO_ROOT / "ci" / "check_ci_image_config.py"
    if not checker.is_file():
        return [f"{checker} is missing, so pyproject.toml cannot be checked"]
    spec = importlib.util.spec_from_file_location("_ci_checker", checker)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The checker addresses its files relative to the repository root, the way
    # it is invoked in CI, so run it from there rather than from wherever this
    # script was started - otherwise it reports pyproject.toml missing, which
    # would read as a problem with the release.
    with contextlib.chdir(REPO_ROOT):
        return module.check_no_direct_references()


def next_milestone_exists(repo, version):
    """There has to be a milestone for the release after this one.

    Every pull request a person opens is given the next open v.YYYYMM
    milestone by .github/workflows/assign_milestone.yml, and the notes for
    that release are generated from it. If the milestone does not exist, the
    pull requests merged after this release belong to nothing and the next
    release note starts empty.
    """
    later = [
        m.title
        for m in repo.get_milestones(state="open")
        if re.fullmatch(r"v\.\d{6}", m.title) and version_of(m.title) > version
    ]
    if later:
        return []
    return [
        f"no open milestone after v.{version}: create the next one (v.YYYYMM) so "
        "pull requests merged from now on land in the next release's notes"
    ]


def merged_without_milestone(client, slug, repo, version):
    """Merged pull requests that no milestone will ever list.

    The notes are "everything in the milestone", so a merged pull request with
    no milestone is invisible to them. Finding those at release time is the
    last chance; assign_milestone.yml is what stops them happening.
    """
    previous = next(
        (t for t in repo.get_tags() if t.name != f"v.{version}"),
        None,
    )
    if previous is None:
        return []
    since = previous.commit.commit.author.date.date().isoformat()
    query = f"repo:{slug} is:pr is:merged no:milestone merged:>={since}"
    try:
        orphans = [
            issue
            for issue in client.search_issues(query)
            if issue.user is None or issue.user.type != "Bot"
        ]
    except github.GithubException:
        return [
            f"could not search for merged pull requests without a milestone ({query})"
        ]
    if not orphans:
        return []
    listed = ", ".join(f"#{i.number}" for i in orphans[:5])
    more = f" and {len(orphans) - 5} more" if len(orphans) > 5 else ""
    return [
        f"{len(orphans)} pull request(s) merged since {previous.name} have no "
        f"milestone, so no release note lists them: {listed}{more}"
    ]


def readme_sections():
    """The What's new list and the Earlier releases list, as text.

    Returns None if README.md is not shaped the way this function edits, so a
    restructured README makes the script say what to do by hand rather than
    rewrite something it does not understand.
    """
    if not README_FILE.is_file():
        return None
    text = README_FILE.read_text()
    match = re.search(
        r"(## What's new\n\n)(.*?)"
        r"(\n<details>\n<summary>Earlier releases</summary>\n\n)(.*?)"
        r"(\nFull history:)",
        text,
        re.S,
    )
    return (text, match) if match else None


def readme_entry(version, where=None):
    """The What's new bullet for this version, or None if there is none.

    `where` is the text to search, so a caller can limit it to the current
    What's new list: a bullet found anywhere else - under Earlier releases,
    or in a README whose sections were renamed - is not this release being
    announced.
    """
    if where is None:
        if not README_FILE.is_file():
            return None
        where = README_FILE.read_text()
    # one bullet: from its "- **[ESPnet <version>]" to the next bullet or blank line
    match = re.search(
        rf"- \*\*\[ESPnet {re.escape(version)}\].*?(?=\n- \*\*\[ESPnet |\n\n|\Z)",
        where,
        re.S,
    )
    return match.group(0) if match else None


def readme_up_to_date(version, will_apply):
    """README's What's new has to name this release, with a real summary.

    The 202610 release went out with What's new still describing 202609: the
    README is the first thing a visitor reads, and nothing in the procedure
    pointed at it. --apply writes the entry, so its absence is excused there,
    the way the version.txt mismatch is.
    """
    sections = readme_sections()
    if sections is None:
        # Not "the entry is missing": the section this reads and writes is not
        # there to read, and --apply would not write it either.
        return [
            "README.md has no What's new section shaped as this script expects "
            "(a list, then <details><summary>Earlier releases</summary>, then "
            "Full history:) - add the entry by hand, or fix the section"
        ]
    entry = readme_entry(version, where=sections[1].group(2))
    if entry is None:
        if will_apply:
            return []
        return [
            f"README.md's What's new does not mention {version} "
            "(--apply writes the entry, then fill in its summary)"
        ]
    if README_PLACEHOLDER in entry:
        return [
            f"README.md's What's new entry for {version} is still the placeholder "
            "- write what the release is about"
        ]
    return []


def update_readme(version):
    """Add the What's new entry for this release and demote the previous one."""
    sections = readme_sections()
    if sections is None:
        print(
            "README.md is not shaped as expected: add the What's new entry for "
            f"{version} by hand"
        )
        return
    text, match = sections
    if readme_entry(version, where=match.group(2)) is not None:
        return
    header, current, opener, earlier, tail = match.groups()
    entry = (
        f"- **[ESPnet {version}]"
        f"(https://github.com/espnet/espnet/releases/tag/v.{version})** —\n"
        f"  {README_PLACEHOLDER}\n"
    )
    demoted = current.strip("\n")
    README_FILE.write_text(
        text[: match.start()]
        + header
        + entry
        + opener
        + (demoted + "\n" if demoted else "")
        + earlier
        + tail
        + text[match.end() :]
    )
    print(f"wrote README.md: What's new now lists {version}")
    print("  replace its placeholder line with what the release is about")


def preflight(client, slug, repo, milestone, version, open_items, will_apply):
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

    problems += readme_up_to_date(version, will_apply)
    problems += next_milestone_exists(repo, version)
    problems += merged_without_milestone(client, slug, repo, version)
    problems += trusted_publishing_ready()
    problems += publishable_metadata()

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

    update_readme(version)

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

    version = release_version(milestone.title)
    if version is None:
        sys.exit(
            f"milestone {milestone.title!r} does not look like a release: expected "
            "v.YYYYMM, optionally with .postN for a patch (PyPI spells it that "
            "way; -patchN is not a version it accepts)"
        )

    grouped, contributors, merged, open_items = collect(repo, milestone)

    problems = preflight(client, slug, repo, milestone, version, open_items, args.apply)
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
        "  1. Write the README What's new summary, and merge it with the\n"
        "     version.txt bump - the same pull request carries both.\n"
        "  2. Fill in the Overview in the draft release.\n"
        f"  3. Publish the draft. That creates tag v.{version}, which triggers\n"
        "     publish_python_package.yml and uploads to PyPI. It cannot be undone.\n"
        "  4. Check the run: gh run list --workflow publish_python_package.yml\n"
        "     Both 202604 tags failed here silently. Do not assume it worked.\n"
        "  5. Move anything still open to the next milestone and close this one.\n"
        "  6. Create the milestone after that, so assign_milestone.yml has\n"
        "     somewhere to put the pull requests opened from now on.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()

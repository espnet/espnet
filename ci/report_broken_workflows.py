#!/usr/bin/env python3
"""Report workflows whose latest run failed on an event nobody watches.

Three failures in this repository went unnoticed for months, and all three had
the same shape: the only path that runs them is a schedule or a release tag, so
their failure never turns a pull request red.

  publish_python_package.yml   5 months. Found by trying to cut a release.
  publish_docker_image.yml     5 months. Found by opening the deployments page.
  k2.done                      4 years. A step that exited 0 having installed
                               nothing; not a workflow, but the same blindness.

So this looks at the latest run of every workflow and reports the ones that
failed, ignoring `pull_request` events - a pull request failing is ordinary and
someone is already looking at it. What is left is exactly the set nobody sees.

Two things learned writing this, both of which cost a wrong answer first:

  Do not query per workflow. `actions/workflows/{id}/runs` for each of 21
  workflows took over nine minutes and half the calls came back empty, which
  reads as "no runs" and hides real failures. Paging `actions/runs` directly
  answers the same question in seconds - and page it by hand, without
  --paginate, or gh follows the Link header from each page to the last and
  fetches the whole list once per page.

  Ignore workflows with no recent run. Disabling a workflow by renaming the file
  (automerge.yml -> automerge.yml.eol) leaves the old entry in the API, still
  carrying whatever its last run concluded - a June failure that is not a
  problem and cannot be fixed.
"""

import argparse
import json
import subprocess
import sys

# Events whose failures are visible to somebody by construction.
WATCHED_BY_A_HUMAN = {"pull_request", "pull_request_target"}


def api(path: str, tries: int = 3):
    """One GET against the GitHub API, retried.

    Without --paginate. Passing it alongside an explicit page= makes gh follow
    the Link header from that page to the last one, so a five-page loop fetches
    the whole list five times; the first version of this took over nine minutes
    that way.
    """
    for _ in range(tries):
        proc = subprocess.run(["gh", "api", path], capture_output=True, text=True)
        if proc.returncode == 0 and proc.stdout.strip():
            try:
                return json.loads(proc.stdout)
            except json.JSONDecodeError:
                continue
    return None


def latest_runs(repo: str, pages: int):
    """The newest run of each workflow, from one pass over the run list."""
    runs = []
    for page in range(1, pages + 1):
        got = api(f"repos/{repo}/actions/runs?per_page=100&page={page}")
        if got is None:
            sys.exit(f"could not read page {page} of {repo}'s runs")
        batch = got.get("workflow_runs", [])
        runs += batch
        if len(batch) < 100:
            break
    newest = {}
    for run in sorted(runs, key=lambda r: r["created_at"]):
        newest[run["path"]] = run
    return newest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="espnet/espnet")
    parser.add_argument(
        "--pages",
        type=int,
        default=5,
        help="pages of 100 runs to scan; enough to cover every workflow's "
        "newest run, not to go back in history",
    )
    args = parser.parse_args()

    newest = latest_runs(args.repo, args.pages)
    broken = [
        run
        for run in newest.values()
        if run.get("conclusion") == "failure"
        and run.get("event") not in WATCHED_BY_A_HUMAN
    ]

    for path, run in sorted(newest.items()):
        state = run.get("conclusion") or run.get("status")
        mark = "FAIL" if run in broken else "    "
        print(
            f"{mark}  {path.split('/')[-1]:<34}{state:<13}"
            f"{run.get('event'):<19}{run['created_at'][:10]}"
        )

    if not broken:
        print(f"\n{len(newest)} workflows, none failing outside pull requests.")
        return 0

    print(f"\n{len(broken)} workflow(s) failing where no pull request shows it:\n")
    for run in broken:
        print(f"  {run['path']}  ({run.get('event')}, {run['created_at'][:10]})")
        print(f"    {run['html_url']}")
    return 1


if __name__ == "__main__":
    sys.exit(main())

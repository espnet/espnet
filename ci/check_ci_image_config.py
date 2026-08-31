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

Third, the config-task lists both the integration and the configuration matrix
are built from must match the tasks their scripts actually implement. A task in
the matrix and not the script runs the script's default and silently tests the
wrong thing; a task in the script and not the matrix is never run at all.

The configuration matrix also shards a task across jobs - "asr:2/3" is the
second third of the asr configs - and a shard set with a gap or a duplicate
would drop or repeat configs with nothing failing to say so, so the shards of
each task must cover 1..n exactly.
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


def implemented(script: Path) -> set:
    """The task names a ci/test_*.sh dispatches on."""
    text = script.read_text()
    return set(re.findall(r'\$\{task\}" == "([a-z0-9_]+)"', text)) - {"all"}


def compare(label: str, wanted: set, have: set) -> list:
    problems = []
    for task in sorted(wanted - have):
        problems.append(f"{label}: {task} is in the matrix but not in the script")
    for task in sorted(have - wanted):
        problems.append(f"{label}: {task} is in the script but never run")
    return problems


def check_integration_tasks() -> list:
    """The integration matrix's task list against what its script implements."""
    workflow = re.search(r"tasks=([a-z0-9_,]+)", CONSUMER.read_text())
    if workflow is None:
        return ["ci_on_ubuntu.yml: no integration tasks= line found"]
    return compare(
        "integration config-task",
        set(workflow.group(1).split(",")),
        implemented(Path("ci/test_integration_espnet2.sh")),
    )


def check_configuration_tasks() -> list:
    """The same for the configuration matrix, which nothing checked before.

    Its list is written inline rather than derived, and a task can carry a
    shard suffix - "asr:2/3" is the second third of the asr configs - so the
    suffix is stripped before comparing. A shard count of 1 is pointless but
    harmless; a task sharded n ways with a gap or a duplicate is not, so the
    shards of each task have to be exactly 1..n.
    """
    match = re.search(r"config-task: \[([^\]]*)\]", CONSUMER.read_text())
    if match is None:
        return ["ci_on_ubuntu.yml: no configuration config-task list found"]
    entries = [item.strip().strip('"') for item in match.group(1).split(",")]

    problems = []
    shards = {}
    wanted = set()
    for entry in entries:
        task, _, spec = entry.partition(":")
        wanted.add(task)
        if not spec:
            continue
        index, _, total = spec.partition("/")
        if not (index.isdigit() and total.isdigit()):
            problems.append(f"configuration config-task {entry}: malformed shard spec")
            continue
        shards.setdefault(task, []).append((int(index), int(total)))

    for task, seen in sorted(shards.items()):
        totals = {total for _, total in seen}
        if len(totals) != 1:
            problems.append(
                f"configuration config-task {task}: disagreeing shard counts {sorted(totals)}"
            )
            continue
        total = totals.pop()
        indexes = sorted(index for index, _ in seen)
        if indexes != list(range(1, total + 1)):
            problems.append(
                f"configuration config-task {task}: shards {indexes} do not cover 1..{total}"
            )

    script = Path("ci/test_configuration_espnet2.sh")
    return problems + compare("configuration config-task", wanted, implemented(script))


def main() -> int:
    bad = check_variants() + check_integration_tasks() + check_configuration_tasks()
    for problem in bad:
        print(problem, file=sys.stderr)
    build, consumer = inputs(BUILD), inputs(CONSUMER)
    if build == consumer and not bad:
        print(
            f"hash inputs agree ({len(build)} entries); every job matrix is a "
            "built variant; integration and configuration tasks match their "
            "scripts, and every shard set is complete"
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

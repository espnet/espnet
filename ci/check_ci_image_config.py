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

Fourth, every step that runs a ci/test_* script must have HF_TOKEN in scope.
Without it the Hugging Face downloads those tests do are anonymous, and
huggingface.co rate limits anonymous callers hard enough to take a whole run
down - it did, with 72 tests failing on 429. For a long time only the install
steps had the token, which is invisible in review because the tests pass
whenever the fleet happens to be under the limit.

Fifth, any third-party action handed a secret must be pinned to a commit SHA.
Not every action - that is a bigger argument - but the ones that receive a
secret, because a mutable ref there means whoever can move the tag can read the
secret. This is not hypothetical: anthropics/claude-code-action@v1 moved between
two resolutions a few hours apart while this check was being written.

Sixth, every codecov upload must pass CODECOV_TOKEN. Without it the upload is
tokenless, which codecov rate limits by IP, and this workflow sends one per job.
The secret has existed since 2024 and was passed to nothing.

And seventh, the workflow files must have no duplicate mapping keys. PyYAML
accepts them and lets the last one win, so writing a second env: block into a
step silently discards the first - which is exactly what nearly dropped
GITHUB_TOKEN from the two steps that need it for torch.hub.
"""

import json
import re
import sys
from pathlib import Path

import yaml

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
        task, colon, spec = entry.partition(":")
        wanted.add(task)
        # An absent colon is a whole task; a colon with nothing useful after it
        # is a mistake. Testing `spec` alone conflated the two, so "asr:" passed
        # here as a bare task and then died in the script instead.
        if not colon:
            continue
        index, slash, total = spec.partition("/")
        if not (slash and index.isdigit() and total.isdigit()):
            problems.append(f"configuration config-task {entry}: malformed shard spec")
            continue
        index, total = int(index), int(total)
        if not 1 <= index <= total:
            problems.append(
                f"configuration config-task {entry}: shard index out of range"
            )
            continue
        shards.setdefault(task, []).append((index, total))

    for task, seen in sorted(shards.items()):
        totals = {total for _, total in seen}
        if len(totals) != 1:
            problems.append(
                f"configuration config-task {task}: disagreeing shard "
                f"counts {sorted(totals)}"
            )
            continue
        total = totals.pop()
        indexes = sorted(index for index, _ in seen)
        if indexes != list(range(1, total + 1)):
            problems.append(
                f"configuration config-task {task}: shards {indexes} "
                f"do not cover 1..{total}"
            )

    script = Path("ci/test_configuration_espnet2.sh")
    return problems + compare("configuration config-task", wanted, implemented(script))


class _NoDuplicates(yaml.SafeLoader):
    """A loader that refuses duplicate mapping keys instead of silently keeping
    the last one. PyYAML's default made a second env: block in a step look
    valid while it discarded the first."""


def _mapping(loader, node, deep=False):
    seen = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in seen:
            raise yaml.YAMLError(
                f"duplicate key {key!r} at line {key_node.start_mark.line + 1}"
            )
        seen.add(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep)


_NoDuplicates.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def _workflows() -> list:
    return sorted(Path(".github/workflows").glob("*.yml"))


def check_no_duplicate_keys() -> list:
    problems = []
    for path in _workflows():
        try:
            yaml.load(path.read_text(), Loader=_NoDuplicates)
        except yaml.YAMLError as error:
            problems.append(f"{path}: {error}")
    return problems


def check_hf_token() -> list:
    """Every step running a ci/test_* script must see HF_TOKEN."""
    problems = []
    for path in _workflows():
        try:
            data = yaml.safe_load(path.read_text())
        except yaml.YAMLError as error:
            problems.append(f"{path}: {error}")
            continue
        for name, job in (data or {}).get("jobs", {}).items():
            if not isinstance(job, dict):
                continue
            job_env = job.get("env") or {}
            for step in job.get("steps") or []:
                if not isinstance(step, dict):
                    continue
                run = str(step.get("run") or "")
                # test_import_all.py only imports modules; it reaches no network
                if "ci/test_" not in run or "test_import_all" in run:
                    continue
                # Both spellings are in use: HF_CI_TOKEN everywhere except the
                # publication job, which has its own HF_TOKEN secret.
                step_env = step.get("env") or {}
                value = step_env.get("HF_TOKEN", job_env.get("HF_TOKEN"))
                if secret_ref(value, "HF_CI_TOKEN", "HF_TOKEN"):
                    continue
                label = step.get("name") or run.strip().split("\n")[0]
                problems.append(
                    f"{path.name}: {name}: step {label!r} runs a test script "
                    "without HF_TOKEN set to a secret"
                )
    return problems


SHA = re.compile(r"[0-9a-f]{40}")
SECRET = re.compile(r"\$\{\{\s*secrets\.([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


def secret_ref(value, *names) -> bool:
    """True when value is a ${{ secrets.NAME }} expression for one of names.

    Testing that the key is merely present is not enough. An empty string, a
    null, or a misspelled secret all leave the job with no token while the key
    is there, and the run then reports success while the upload is anonymous.
    """
    if not isinstance(value, str):
        return False
    match = SECRET.fullmatch(value.strip())
    return bool(match) and match.group(1) in names


def check_secret_actions_pinned() -> list:
    """Any third-party action receiving a secret must be pinned to a SHA.

    A tag can be moved by whoever controls the action's repository, so a secret
    passed to a mutable ref is readable by whatever that ref points at next.
    """
    problems = []
    for path in _workflows():
        try:
            data = yaml.safe_load(path.read_text())
        except yaml.YAMLError:
            continue  # check_no_duplicate_keys reports the parse failure
        for name, job in (data or {}).get("jobs", {}).items():
            if not isinstance(job, dict):
                continue
            for step in job.get("steps") or []:
                if not isinstance(step, dict):
                    continue
                uses = str(step.get("uses") or "")
                # local composite actions carry no ref and cannot be moved
                if "/" not in uses or "@" not in uses or uses.startswith("./"):
                    continue
                passed = json.dumps({"with": step.get("with"), "env": step.get("env")})
                if "secrets." not in passed:
                    continue
                if SHA.fullmatch(uses.rsplit("@", 1)[1]):
                    continue
                problems.append(
                    f"{path.name}: {name}: {uses} receives a secret but is not "
                    "pinned to a commit SHA"
                )
    return problems


def check_codecov_token() -> list:
    """Every codecov-action step must pass the token.

    Tokenless uploads are rate limited by IP, and nothing fails when one is
    dropped - the coverage just quietly does not arrive.
    """
    problems = []
    for path in _workflows():
        try:
            data = yaml.safe_load(path.read_text())
        except yaml.YAMLError:
            continue  # check_no_duplicate_keys reports the parse failure
        for name, job in (data or {}).get("jobs", {}).items():
            if not isinstance(job, dict):
                continue
            for step in job.get("steps") or []:
                if not isinstance(step, dict):
                    continue
                if "codecov/codecov-action" not in str(step.get("uses") or ""):
                    continue
                token = (step.get("with") or {}).get("token")
                if secret_ref(token, "CODECOV_TOKEN"):
                    continue
                problems.append(
                    f"{path.name}: {name}: codecov upload without "
                    "token: ${{ secrets.CODECOV_TOKEN }}"
                )
    return problems


def main() -> int:
    bad = (
        check_variants()
        + check_integration_tasks()
        + check_configuration_tasks()
        + check_no_duplicate_keys()
        + check_hf_token()
        + check_secret_actions_pinned()
        + check_codecov_token()
    )
    for problem in bad:
        print(problem, file=sys.stderr)
    build, consumer = inputs(BUILD), inputs(CONSUMER)
    if build == consumer and not bad:
        print(
            f"hash inputs agree ({len(build)} entries); every job matrix is a "
            "built variant; integration and configuration tasks match their "
            "scripts; every shard set is complete; every test step has "
            "HF_TOKEN; every action given a secret is pinned; every codecov "
            "upload has a token; no duplicate keys"
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

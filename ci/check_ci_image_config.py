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

Fifth, every third-party action must be pinned to a commit SHA - in the
composite actions under .github/actions as well as in the workflows. A tag or a
branch can be moved by whoever controls the action's repository, and the step
then runs code nobody here reviewed. This is not hypothetical:
anthropics/claude-code-action@v1 moved between two resolutions a few hours apart
while this check was being written.

Sixth, every codecov upload must pass CODECOV_TOKEN. Without it the upload is
tokenless, which codecov rate limits by IP, and this workflow sends one per job.
The secret has existed since 2024 and was passed to nothing.

Every job must also have a permissions block in scope, at the workflow or the
job level. Inheriting the repository default means inheriting write on almost
everything, and nothing fails to say so.

Seventh, pyproject.toml must declare no direct references - a dependency
written "name @ git+https://..." rather than as a version range. PyPI refuses any
distribution whose metadata contains one, with 400 Can't have direct dependency,
and nothing else in the repository looks: the build succeeds, every test passes,
and the failure surfaces only when a release tag is pushed. Three of them arrived
with the setup.py to pyproject.toml migration and went unnoticed for five months,
because the release before it had deliberately moved the same three packages out
to tools/Makefile and nothing recorded why.

Eighth, every pytorch version named outside ci/image_variants.json must be one
that file lists. tools/Makefile's TH_VERSION default and the docker publish
workflow's build argument are the two places that name one, and both are only
reached by paths no pull request exercises - the Makefile default because
ci/install.sh always passes TH_VERSION, the workflow because it runs on a
schedule. So when install_torch.sh stopped supporting 2.7.1, the default stayed
at 2.7.1 and the weekly docker publish failed every Monday for five months
without a single red pull request.

And ninth, the workflow files must have no duplicate mapping keys. PyYAML
accepts them and lets the last one win, so writing a second env: block into a
step silently discards the first - which is exactly what nearly dropped
GITHUB_TOKEN from the two steps that need it for torch.hub.
"""

import json
import re
import sys
import tomllib
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


def _action_files() -> list:
    """Workflows plus the composite actions, which also carry `uses:` steps.

    prepare-environment/action.yml has one, and globbing only .github/workflows
    would leave it unchecked.
    """
    return _workflows() + sorted(Path(".github/actions").glob("*/action.yml"))


def _steps(data) -> list:
    """(container name, step) for a workflow's jobs or a composite's runs."""
    out = []
    for name, job in (data or {}).get("jobs", {}).items():
        if isinstance(job, dict):
            out += [(name, s) for s in (job.get("steps") or [])]
    runs = (data or {}).get("runs")
    if isinstance(runs, dict):
        out += [("runs", s) for s in (runs.get("steps") or [])]
    return out


def check_no_duplicate_keys() -> list:
    # _action_files(), not _workflows(): check_actions_pinned reads the composite
    # actions too, and its `continue` on a parse failure would swallow the error
    # unless something else reports it. That something is this.
    problems = []
    for path in _action_files():
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


def check_actions_pinned() -> list:
    """Every third-party action must be pinned to a commit SHA.

    A tag or a branch can be moved by whoever controls the action's repository.
    The step then runs code nobody here reviewed, with whatever the job holds -
    a secret where one is passed, and the workspace either way.
    """
    problems = []
    for path in _action_files():
        try:
            data = yaml.safe_load(path.read_text())
        except yaml.YAMLError:
            continue  # check_no_duplicate_keys reports the parse failure
        for name, step in _steps(data):
            if not isinstance(step, dict):
                continue
            uses = str(step.get("uses") or "")
            # local composite actions carry no ref and cannot be moved
            if "/" not in uses or "@" not in uses or uses.startswith("./"):
                continue
            if SHA.fullmatch(uses.rsplit("@", 1)[1]):
                continue
            problems.append(
                f"{path.name}: {name}: {uses} is not pinned to a commit SHA"
            )
    return problems


def check_checkout_credentials() -> list:
    """actions/checkout must not leave GITHUB_TOKEN in .git/config.

    The default writes it there, and these jobs then run checked-out pull
    request code. Nothing here needs it: .github/ contains no git push, commit,
    tag or submodule, and peaceiris/actions-gh-pages takes its github_token as
    an explicit input.

    Exempt: a job holding contents: write, which is how a job that is meant to
    push says so. claude.yml is the only one, and its whole purpose is to commit.
    """
    problems = []
    for path in _action_files():
        try:
            data = yaml.safe_load(path.read_text())
        except yaml.YAMLError:
            continue  # check_no_duplicate_keys reports the parse failure
        jobs = (data or {}).get("jobs", {})
        for name, job in jobs.items():
            if not isinstance(job, dict):
                continue
            if (job.get("permissions") or {}).get("contents") == "write":
                continue
            for step in job.get("steps") or []:
                if not isinstance(step, dict):
                    continue
                if "actions/checkout@" not in str(step.get("uses") or ""):
                    continue
                if (step.get("with") or {}).get("persist-credentials") is False:
                    continue
                problems.append(
                    f"{path.name}: {name}: checkout without "
                    "persist-credentials: false"
                )
    return problems


def check_permissions_declared() -> list:
    """Every job must have permissions in scope, at workflow or job level.

    Silence means the repository default, which is write on almost everything -
    see #6583, where a run's own dump showed Contents: write, Actions: write and
    PullRequests: write on jobs that only read.
    """
    problems = []
    for path in _workflows():
        try:
            data = yaml.safe_load(path.read_text())
        except yaml.YAMLError:
            continue  # check_no_duplicate_keys reports the parse failure
        workflow_level = (data or {}).get("permissions")
        for name, job in (data or {}).get("jobs", {}).items():
            if not isinstance(job, dict):
                continue
            if job.get("permissions") is not None or workflow_level is not None:
                continue
            problems.append(
                f"{path.name}: {name}: no permissions in scope, so it inherits "
                "the repository default"
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


PYPROJECT = Path("pyproject.toml")
# PEP 508 calls the "name @ <url>" form a direct reference. Match on the
# separator rather than on "git+", so a plain https:// archive or a file:// path
# is caught too - PyPI rejects every direct reference, not only git ones.
DIRECT_REF = re.compile(r"^[A-Za-z0-9._-]+\s*(\[[^\]]*\])?\s*@\s*\S+")


def declared_dependencies() -> list:
    """Every dependency string in pyproject.toml, with the table it came from.

    Parsed with tomllib, not scanned line by line. A scan for lines that begin
    with a quote misses an array written on one line -

        dependencies = ["example @ git+https://example.invalid/x.git"]

    - because that line begins with the key. The first version of this check did
    exactly that, so it would have passed a pyproject no release could publish:
    the same defect as check_codecov_token testing for the presence of a token
    key rather than its value.
    """
    data = tomllib.loads(PYPROJECT.read_text())
    project = data.get("project") or {}
    found = [("project.dependencies", d) for d in project.get("dependencies") or []]
    for extra, items in (project.get("optional-dependencies") or {}).items():
        table = f"project.optional-dependencies.{extra}"
        found += [(table, item) for item in items or []]
    return found


def check_no_direct_references() -> list:
    """No declared dependency may be a PEP 508 direct reference."""
    if not PYPROJECT.exists():
        return [f"{PYPROJECT}: missing"]
    lines = PYPROJECT.read_text().splitlines()
    problems = []
    for table, entry in declared_dependencies():
        if not DIRECT_REF.match(entry.strip()):
            continue
        # Only to point at the offender; tomllib is what decided it is one.
        number = next((n for n, ln in enumerate(lines, 1) if entry in ln), None)
        at = f":{number}" if number else ""
        problems.append(
            f"{PYPROJECT}{at}: direct reference in [{table}]: {entry}\n"
            "  PyPI rejects any distribution whose metadata contains one "
            "(400 Can't have direct dependency), so this makes every release "
            "upload fail, and nothing before the tag says so.\n"
            "  Depend on a published version instead, the way espnet-g2p-en, "
            "espnet-ctc-segmentation and espnet-s3prl are."
        )
    return problems


INSTALL_TORCH = Path("tools/installers/install_torch.sh")

# Everywhere outside ci/image_variants.json that names a python or pytorch
# version. An explicit list rather than a scan of every file: a scan invents
# false positives on comments and prose, and this is meant to be readable.
VARIANT_SITES = (
    (Path("tools/Makefile"), re.compile(r"^TH_VERSION\s*:?=\s*(\S+)", re.M), "pytorch"),
    (
        Path("docker/ci.dockerfile"),
        re.compile(r"^ARG PYTHON_VERSION=(\S+)", re.M),
        "python",
    ),
    (
        Path("docker/ci.dockerfile"),
        re.compile(r"^ARG TH_VERSION=(\S+)", re.M),
        "pytorch",
    ),
    (
        Path("docker/prebuilt/devel.dockerfile"),
        re.compile(r"conda install[^\n]*\"python=([0-9.]+)\""),
        "python",
    ),
    (
        Path(".github/workflows/publish_docker_image.yml"),
        re.compile(r"--build-arg\s+TH_VERSION=(\S+)"),
        "pytorch",
    ),
)


def check_versions_are_built_variants() -> list:
    """Every version named at those sites must be one image_variants.json lists.

    install_torch.sh exits 1 on a pytorch version outside that set, and the
    python floor is a hard requirement of pyproject.toml, so a version outside
    it is not a slow path - it is a build that cannot succeed. Two of these
    sites were wrong for five months (TH_VERSION 2.7.1 and conda python=3.11)
    because the only path that reads them is the weekly docker publish, which
    runs on a schedule and so never turns a pull request red.
    """
    pythons, torches = variants()
    allowed = {"python": pythons, "pytorch": torches}
    problems = []
    for path, pattern, axis in VARIANT_SITES:
        if not path.exists():
            problems.append(f"{path}: missing, so its version cannot be checked")
            continue
        text = path.read_text()
        for match in pattern.finditer(text):
            found = match.group(1).strip("\"'")
            if found in allowed[axis]:
                continue
            line = text[: match.start()].count("\n") + 1
            problems.append(
                f"{path}:{line}: names {axis} {found}, which "
                f"ci/image_variants.json does not list "
                f"({', '.join(allowed[axis])})"
            )
    return problems


def _order(version: str) -> tuple:
    """Sort key for a dotted version, so "3.9" does not outrank "3.12"."""
    return tuple(int(part) for part in version.split("."))


def check_declared_support_matches_variants() -> list:
    """What the package tells the world must be the set CI actually tests."""
    pythons, torches = variants()
    problems = []

    text = PYPROJECT.read_text()
    classifiers = set(
        re.findall(r'"Programming Language :: Python :: (\d+\.\d+)"', text)
    )
    if classifiers != set(pythons):
        problems.append(
            f"{PYPROJECT}: python classifiers are {sorted(classifiers)}, but "
            f"ci/image_variants.json builds {sorted(pythons)}"
        )

    match = re.search(r'requires-python\s*=\s*"([^"]+)"', text)
    floor = re.search(r">=\s*(\d+\.\d+)", match.group(1)) if match else None
    lowest = min(pythons, key=_order)
    if floor is None or floor.group(1) != lowest:
        got = floor.group(1) if floor else match.group(1) if match else "nothing"
        problems.append(
            f"{PYPROJECT}: requires-python floors python at {got}, but the "
            f"lowest version CI builds is {lowest}"
        )

    match = re.search(r'"torch>=([0-9.]+)', text)
    lowest_torch = min(torches, key=_order)
    if match is None or match.group(1) != lowest_torch:
        got = match.group(1) if match else "nothing"
        problems.append(
            f"{PYPROJECT}: torch is floored at {got}, but the lowest version "
            f"CI builds is {lowest_torch}"
        )

    if not INSTALL_TORCH.exists():
        problems.append(f"{INSTALL_TORCH}: missing")
        return problems
    installable = set(
        re.findall(
            r"^\s*install_torch (\d+\.\d+\.\d+)",
            INSTALL_TORCH.read_text(),
            re.M,
        )
    )
    if installable != set(torches):
        problems.append(
            f"{INSTALL_TORCH}: installs {sorted(installable)}, but "
            f"ci/image_variants.json builds {sorted(torches)}. It exits 1 on "
            "anything it does not install"
        )
    return problems


def main() -> int:
    bad = (
        check_variants()
        + check_integration_tasks()
        + check_configuration_tasks()
        + check_no_duplicate_keys()
        + check_hf_token()
        + check_actions_pinned()
        + check_checkout_credentials()
        + check_permissions_declared()
        + check_codecov_token()
        + check_no_direct_references()
        + check_versions_are_built_variants()
        + check_declared_support_matches_variants()
    )
    for problem in bad:
        print(problem, file=sys.stderr)
    build, consumer = inputs(BUILD), inputs(CONSUMER)
    if build == consumer and not bad:
        print(
            f"hash inputs agree ({len(build)} entries); every job matrix is a "
            "built variant; integration and configuration tasks match their "
            "scripts; every shard set is complete; every test step has "
            "HF_TOKEN; every third-party action is pinned to a SHA; "
            "every checkout drops its credentials; every job declares "
            "permissions; every codecov upload has a token; pyproject declares "
            "no direct references; every python and pytorch version named "
            "outside image_variants.json is one it lists, and what the "
            "package declares matches it; no duplicate keys"
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

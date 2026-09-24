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

Eighth, every python or pytorch version pinned anywhere in the repository must
be one ci/image_variants.json lists. Almost none of these sites is reached by a
path a pull request exercises - tools/Makefile's default because ci/install.sh
always passes TH_VERSION, the docker publish workflow because it runs on a
schedule, the rest because they are instructions for a human - so a rotted one
stays green until someone follows it. That is how the weekly docker publish
failed every Monday for five months.

This check used to hold a list of the files that pin a version, and the list was
the bug: docker/build.sh, both devcontainer files, tools/setup_uv.sh and the
build instructions in doc/installation.md were not in it, and all four sat on
versions install_torch.sh had long stopped accepting. They were found by a
by-hand grep of the whole repository, which is not a check, so the list is gone
- every tracked file is scanned for the shapes that pin a version, and the
handful of version strings that are records rather than instructions are named
in ALLOWED_PINS.

Tenth, every download in the CI and installer scripts must survive a transient
failure. Almost all of them used to pass --tries=3, and that flag does not cover
an HTTP error response at all: against a server returning 500, wget --tries=3
makes exactly one request and exits 8. Nor does it start a fresh connection,
which is what a failed TLS handshake needs. Asking for retries and getting none
is worse than not asking, because the flag is right there in review.
github.com answered one 500 for the miniforge installer and took the macOS
install job down with it. So a shell script downloads through
download_with_retry, and the two dockerfiles and the two `curl | bash` pipes,
which cannot source it, carry --retry-on-http-error with the full transient set
(a partial list retries only what it names) or curl's --fail with --retry N.
curl needs --fail specifically: without it a 500 is not an error, so curl exits
0 and writes the error body to the output - which, piped to a shell, runs it.

Eleventh, tools/installers/install_k2.sh names the torch versions k2 has no
wheel for yet, so that they skip k2 rather than fail the image build, and every
version it names must be one ci/image_variants.json builds. The k2 tests are
pytest.importorskip, so that skip never turns a run red on its own, and a stale
entry would keep k2 out of a variant that could have had it.

And ninth, the workflow files must have no duplicate mapping keys. PyYAML
accepts them and lets the last one win, so writing a second env: block into a
step silently discards the first - which is exactly what nearly dropped
GITHUB_TOKEN from the two steps that need it for torch.hub.
"""

import fnmatch
import importlib.util
import json
import re
import subprocess
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


VARIANTS_SCRIPT = Path("ci/image_variants.py")
# Every `image_variants.py matrix ...` the workflow runs, with the shell
# variables left in - they are dropped before the command is re-run here.
GENERATED = re.compile(r"python3 ci/image_variants\.py matrix([^\n)]*)")


def _k2_gap() -> set:
    """Torch versions install_k2.sh skips k2 for."""
    if not INSTALL_K2.exists():
        return set()
    match = re.search(r'^k2_missing_for="([^"]*)"', INSTALL_K2.read_text(), re.M)
    return set(match.group(1).split()) if match else set()


RELEVANCE = Path("ci/integration_is_relevant.py")
# Removing any of these makes pull requests that change them stop running the
# recipe tests, which is the failure worth guarding: it is green and faster,
# and the only thing that catches it is the master push after the merge. The
# rest of the list is judgement and can be edited freely.
RELEVANCE_CORE = {
    "espnet2": (
        "espnet2/",
        "egs2/TEMPLATE/",
        "egs2/mini_an4/",
        "tools/",
        "ci/test_integration_espnet2.sh",
    ),
    # espnet2/ is here because espnet3 imports it throughout, so an espnet2
    # change can break the espnet3 recipes without touching espnet3.
    "espnet3": (
        "espnet2/",
        "espnet3/",
        "egs3/",
        "tools/",
        "ci/test_integration_espnet3.sh",
    ),
}


REPORTER = Path(".github/workflows/report_broken_workflows.yml")
# How long after the last other daily schedule the reporter must run. GitHub
# does not start a scheduled run on time - this repository has seen a 06:00
# cron fire at 06:29 - so "later" needs a margin, not a minute.
REPORTER_MARGIN = 60


def _daily_crons() -> tuple:
    """({workflow path: [minutes past midnight]}, [what could not be read]).

    Anything daily-shaped that this cannot evaluate goes in the second list
    rather than being dropped. Silently skipping what it cannot parse is how
    a rule like this passes while the thing it guards is broken - a `.yaml`
    file, a timezone, or `0 8,10 * * *` would each have done it.
    """
    found, unreadable = {}, []
    files = sorted(
        list(Path(".github/workflows").glob("*.yml"))
        + list(Path(".github/workflows").glob("*.yaml"))
    )
    for path in files:
        try:
            workflow = yaml.safe_load(path.read_text())
        except yaml.YAMLError as e:
            unreadable.append(f"{path}: not valid YAML ({e.__class__.__name__})")
            continue
        if not isinstance(workflow, dict):
            continue
        # `on` is the YAML 1.1 boolean True once parsed
        triggers = workflow.get("on") or workflow.get(True) or {}
        if not isinstance(triggers, dict):
            continue
        times = []
        for entry in triggers.get("schedule") or []:
            if not isinstance(entry, dict):
                unreadable.append(f"{path}: a schedule entry that is not a mapping")
                continue
            fields = str(entry.get("cron", "")).split()
            if len(fields) != 5:
                unreadable.append(
                    f"{path}: cron {entry.get('cron')!r} is not five fields"
                )
                continue
            if fields[2:] != ["*", "*", "*"]:
                continue  # weekly or monthly; not part of this ordering
            zone = entry.get("timezone")
            if zone:
                # GitHub takes an IANA timezone here, and its UTC time then
                # moves with daylight saving. Rather than guess a fixed
                # offset, say so: there are none today, and one added later
                # deserves a decision rather than a silent pass.
                unreadable.append(
                    f"{path}: cron {entry.get('cron')!r} has timezone {zone!r}, "
                    "which this check cannot convert to UTC"
                )
                continue
            try:
                minute, hour = int(fields[0]), int(fields[1])
            except ValueError:
                # a list, range or step - `0 8,10 * * *` is daily and runs at
                # two times, one of which may be after the reporter
                unreadable.append(
                    f"{path}: cron {entry.get('cron')!r} is daily but its hour "
                    "or minute is not a plain number, so this check cannot "
                    "tell when it runs"
                )
                continue
            times.append(hour * 60 + minute)
        if times:
            found[path] = times
    return found, unreadable


def check_reporter_runs_last() -> list:
    """The broken-workflow report must be the last daily schedule.

    It reports the latest run of every workflow, so anything scheduled after
    it is reported a day late - which is not a wrong answer, but it is a stale
    one, and it reads as a live failure. At 06:00 it was filing
    check_demo_links (07:00) results from the previous morning, and issue
    #6800 named a demo-link failure that had already passed.
    """
    crons, unreadable = _daily_crons()
    problems = [
        f"{where}.\n  A daily schedule this check cannot place is a daily "
        "schedule it cannot prove runs before the report."
        for where in unreadable
    ]
    mine = crons.get(REPORTER)
    if not mine:
        return problems + [
            f"{REPORTER}: no daily cron, so it cannot be checked to run last"
        ]
    others = {path: max(times) for path, times in crons.items() if path != REPORTER}
    if not others:
        return problems
    latest_path, latest = max(others.items(), key=lambda kv: kv[1])
    if min(mine) < latest + REPORTER_MARGIN:
        return problems + [
            f"{REPORTER}: runs at {min(mine) // 60:02d}:{min(mine) % 60:02d} UTC, "
            f"but {latest_path.name} runs at {latest // 60:02d}:{latest % 60:02d} "
            f"and it needs at least {REPORTER_MARGIN} minutes after the last "
            "other daily schedule.\n"
            "  It reports each workflow's latest run, so one scheduled after it "
            "is reported a day late - a failure that has already been fixed, "
            "and a fresh one missed for a day."
        ]
    return problems


def check_needed_before_read() -> list:
    """A job reading another job's outputs must declare it in `needs`.

    An expression naming a job that is not a dependency evaluates to the empty
    string rather than failing, so the step sees "" and carries on. Writing
    `needs.resolve_ci_image.outputs.espnet3_relevant` into a job whose needs
    was only process_labels is how the espnet3 publication test came within
    one commit of being skipped on every pull request, silently and in green.
    """
    workflow = yaml.safe_load(CONSUMER.read_text())
    problems = []
    for name, job in (workflow.get("jobs") or {}).items():
        needs = job.get("needs") or []
        if isinstance(needs, str):
            needs = [needs]
        for read in sorted(
            set(re.findall(r"needs\.([a-z_0-9]+)\.outputs", yaml.dump(job)))
        ):
            if read not in needs:
                problems.append(
                    f"{CONSUMER}: job {name} reads "
                    f"needs.{read}.outputs but does not need {read}, so the "
                    "expression is the empty string rather than an error"
                )
    return problems


def check_integration_relevance_paths() -> list:
    """Every path prefix that gates the integration tests must still exist.

    A prefix that matches nothing is a path that was renamed or removed, and
    it fails silently in the dangerous direction: pull requests touching what
    used to live there stop running the recipe tests, and the jobs go green
    faster, which looks like the change working.
    """
    if not RELEVANCE.exists():
        return [f"{RELEVANCE}: missing"]
    spec = importlib.util.spec_from_file_location("relevance", RELEVANCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Not tracked_files(): that one is for the version-pin scan and skips the
    # recipes, which is most of what this list is about.
    listed = subprocess.run(
        ["git", "ls-files", "-z"], capture_output=True, text=True, check=False
    )
    if listed.returncode != 0:
        return [f"{RELEVANCE}: could not list tracked files"]
    paths = [name for name in listed.stdout.split("\0") if name]
    problems = []
    for suite, prefixes in module.RELEVANT.items():
        problems += [
            f"{RELEVANCE}: [{suite}] the prefix {prefix!r} matches no tracked "
            "file, so whatever used to be there no longer runs the "
            "integration tests"
            for prefix in prefixes
            if not any(name == prefix or name.startswith(prefix) for name in paths)
        ]
        problems += [
            f"{RELEVANCE}: [{suite}] {prefix!r} is missing, so a pull request "
            "that changes it would not run these integration tests at all"
            for prefix in RELEVANCE_CORE[suite]
            if prefix not in prefixes
        ]
    for suite in RELEVANCE_CORE:
        if suite not in module.RELEVANT:
            problems.append(f"{RELEVANCE}: no list for the {suite} suite")
    return problems


def check_generated_matrices() -> list:
    """A matrix the workflow generates must still cover every python.

    The integration grid is narrowed to one pytorch per python on pull
    requests, because 14 tasks across the full grid is 84 jobs and the whole
    critical path. Narrowing the other axis instead would drop the axis that
    actually catches things - every version-specific integration failure in
    300 runs was specific to a python - and nothing else would say so: the
    jobs would pass, in half the time, testing half of what they claim to.
    """
    pythons, pytorches = variants()
    problems = []
    for match in GENERATED.finditer(CONSUMER.read_text()):
        # Shell expansions cannot be evaluated here; the flags can.
        words = [w.strip("\"'") for w in match.group(1).split()]
        arguments = [w for w in words if w and "${" not in w]
        run = subprocess.run(
            [sys.executable, str(VARIANTS_SCRIPT), "matrix", *arguments],
            capture_output=True,
            text=True,
            check=False,
        )
        shown = " ".join(arguments) or "(no arguments)"
        if run.returncode != 0:
            problems.append(
                f"{CONSUMER}: `image_variants.py matrix {shown}` exits "
                f"{run.returncode}: {run.stderr.strip()[:120]}"
            )
            continue
        try:
            grid = json.loads(run.stdout)
        except json.JSONDecodeError:
            problems.append(
                f"{CONSUMER}: `image_variants.py matrix {shown}` did not print JSON"
            )
            continue
        if grid.get("python-version") != list(pythons):
            problems.append(
                f"{CONSUMER}: `image_variants.py matrix {shown}` yields pythons "
                f"{grid.get('python-version')}, but ci/image_variants.json "
                f"builds {list(pythons)}.\n"
                "  The python axis is the one that catches things - every "
                "version-specific integration failure in 300 runs was specific "
                "to a python and failed on every pytorch. Narrow pytorch, "
                "never python."
            )
        unbuilt = [v for v in grid.get("pytorch-version", []) if v not in pytorches]
        if unbuilt:
            problems.append(
                f"{CONSUMER}: `image_variants.py matrix {shown}` asks for "
                f"pytorch {unbuilt}, which no image is built for"
            )
        if not grid.get("pytorch-version"):
            problems.append(
                f"{CONSUMER}: `image_variants.py matrix {shown}` yields no pytorch"
            )
        # A narrowed grid must not land on a version the suite only half
        # supports. install_k2.sh skips k2 for those, and the k2 blocks in
        # ci/test_integration_espnet2.sh are guarded by `import k2`, so a
        # pull-request grid pinned there would stop exercising them anywhere
        # except master - green, faster, and testing less than it says.
        if len(grid.get("pytorch-version", [])) < len(pytorches):
            gap = _k2_gap() & set(grid["pytorch-version"])
            if gap:
                problems.append(
                    f"{CONSUMER}: `image_variants.py matrix {shown}` narrows to "
                    f"pytorch {sorted(gap)}, which {INSTALL_K2} lists in "
                    "k2_missing_for.\n"
                    "  On that version k2 is not installed and the k2 parts of "
                    "the suite skip themselves, so narrowing onto it stops "
                    "running them on pull requests at all."
                )
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

# Every way this repository pins a python or pytorch version, as patterns
# applied to every tracked file rather than as a list of the files that do it.
# The list came first, and the list was the bug: it named tools/Makefile, two
# dockerfile ARGs and the docker publish workflow, and for as long as it did,
# docker/build.sh sat on torch 2.8.0, both devcontainer files on 2.7.1,
# tools/setup_uv.sh installed 2.6.0, and doc/installation.md told a new user to
# build with 1.10.1 - none of them a version install_torch.sh still accepts, and
# none of them red, because a file nobody added is a file nobody checks. Finding
# them took a by-hand grep of the whole repository, which is not a check.
#
# A scan does invent false positives on prose, which is what the list was for,
# so these match the shapes that pin a version and nothing else - an assignment,
# a pip specifier, a matrix entry - and the real exceptions are named in
# ALLOWED_PINS with a reason each.
LINE_PINS = (
    (re.compile(r"\bTH_VERSION\s*[:=]{1,2}\s*[\"']?(\d+\.\d+\.\d+)"), "pytorch"),
    (re.compile(r"\bth_ver\s*=\s*[\"']?(\d+\.\d+\.\d+)"), "pytorch"),
    # \s* around the operators: a specifier written with spaces around == or >=
    # is the same pin as the unspaced form, and a scan that misses one shape is
    # the list problem again in miniature. (Spelling an example here would trip
    # the scan itself, which is the check working.)
    (re.compile(r"\btorch\s*==\s*(\d+\.\d+\.\d+)"), "pytorch"),
    (re.compile(r"[\"']torch\s*>=\s*(\d+\.\d+\.\d+)"), "pytorch"),
    (re.compile(r"\bPYTHON_VERSION\s*[:=]{1,2}\s*[\"']?(\d+\.\d+)"), "python"),
    (re.compile(r"conda install[^\n]*\"python=(\d+\.\d+)\""), "python"),
    (re.compile(r"uv venv -p (\d+\.\d+)"), "python"),
)

# A matrix line names any number of versions: `pytorch-version: [a, b, c]`.
MATRIX_PINS = (
    ("pytorch-version:", re.compile(r"\d+\.\d+\.\d+"), "pytorch"),
    ("python-version:", re.compile(r"\d+\.\d+(?!\.\d)"), "python"),
)

# Version strings that are records rather than instructions: what a recipe was
# run on, and npm lockfiles.
PIN_SCAN_SKIP = ("egs/", "egs2/", "egs3/", "doc/vuepress/")

ALLOWED_PINS = frozenset(
    {
        # The issue templates carry a filled-in example of the reporter's own
        # environment, from whenever they were written. Nothing is installed
        # from them, and bumping them would say nothing to anyone.
        (".github/ISSUE_TEMPLATE/bug_report.md", "1.4.0"),
        (".github/ISSUE_TEMPLATE/installation-issue-template.md", "1.4.0"),
        (".github/ISSUE_TEMPLATE/installation-issue-template.md", "1.3.1"),
    }
)


def tracked_files() -> list:
    """Every tracked file the pin scan looks at, skipping records and .eol."""
    listed = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, check=True
    ).stdout.split("\n")
    return [
        name
        for name in listed
        if name and not name.startswith(PIN_SCAN_SKIP) and not name.endswith(".eol")
    ]


README = Path("README.md")
CONTRIBUTING = Path("CONTRIBUTING.md")
# The two CI tables and where each one lives. The badge grid stays in the
# README's "Tested environments"; the table saying what each column covers sits
# in CONTRIBUTING 5.3, beside the rest of the testing sections, because it is
# what someone whose pull request just went red is looking for. Both carry the
# pytorch list as column headers and both have to keep naming the real grid,
# wherever they are - so this maps prefix to file rather than assuming one.
COVERAGE_TABLES = {
    "|system/pytorch ver.|": README,
    "|test suite|": CONTRIBUTING,
}
SUITE_HEADER = "|test suite|"
K2_ROW = "|k2-dependent tests|"
# The rows whose "runs on a pull request" columns must be the narrowed grid,
# so that changing which pytorch a pull request gets cannot leave the table
# describing the old one.
PR_ROWS = ("|`espnet2` recipe integration|", "|`espnet3` integration|")


def _columns(line: str) -> list:
    return [cell.strip() for cell in line.strip().strip("|").split("|")][1:]


def check_coverage_tables() -> list:
    """The CI tables must name exactly the grid, and agree about k2.

    The existing version check rejects a version the grid does not build, which
    catches a stale column but not a missing one: add a pytorch and the tables
    quietly describe a grid one column smaller than the real one. And the k2
    row is a written-down copy of k2_missing_for, which is the fact that has
    already gone stale once.

    Each table is looked for in the file that holds it, so moving one between
    documents is a one-line change here rather than a check that starts
    reporting a missing table.
    """
    torches = list(variants()[1])
    problems = []
    text = {}
    for path in dict.fromkeys(COVERAGE_TABLES.values()):
        if not path.exists():
            problems.append(f"{path}: missing")
            continue
        text[path] = path.read_text(encoding="utf-8").splitlines()

    headers = {}
    for prefix, path in COVERAGE_TABLES.items():
        lines = text.get(path)
        if lines is None:
            continue
        found = False
        for number, line in enumerate(lines, 1):
            if not line.startswith(prefix):
                continue
            found = True
            headers[prefix] = (path, number, _columns(line))
            if _columns(line) != torches:
                problems.append(
                    f"{path}:{number}: the table headed {prefix!r} lists "
                    f"pytorch {_columns(line)}, but ci/image_variants.json "
                    f"builds {torches}"
                )
        if not found:
            problems.append(f"{path}: no table headed {prefix!r}")

    narrowed = json.loads(
        subprocess.run(
            [sys.executable, str(VARIANTS_SCRIPT), "matrix", "--newest-pytorch"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
        or "{}"
    ).get("pytorch-version", [])
    suite = headers.get(SUITE_HEADER)
    if suite is not None:
        suite_path, _, columns = suite
        for prefix in PR_ROWS:
            rows = [
                (number, line)
                for number, line in enumerate(text[suite_path], 1)
                if line.startswith(prefix)
            ]
            if not rows:
                problems.append(f"{suite_path}: no row starting {prefix!r}")
                continue
            for number, line in rows:
                cells = _columns(line)
                if len(cells) != len(columns):
                    problems.append(
                        f"{suite_path}:{number}: {prefix} has {len(cells)} cells "
                        f"for {len(columns)} pytorch columns"
                    )
                    continue
                on_pr = [v for v, cell in zip(columns, cells) if "PR" in cell]
                if on_pr != narrowed:
                    problems.append(
                        f"{suite_path}:{number}: {prefix} says a pull request "
                        f"runs pytorch {on_pr}, but image_variants.py "
                        f"--newest-pytorch gives {narrowed}"
                    )

    gap = _k2_gap()
    k2_path = COVERAGE_TABLES[SUITE_HEADER]
    seen_k2_row = False
    for number, line in enumerate(text.get(k2_path, []), 1):
        if not line.startswith(K2_ROW):
            continue
        seen_k2_row = True
        if suite is None:
            break
        columns = suite[2]
        cells = _columns(line)
        if len(cells) != len(columns):
            problems.append(
                f"{k2_path}:{number}: the k2 row has {len(cells)} "
                f"cells for {len(columns)} pytorch columns"
            )
            break
        said = {v for v, cell in zip(columns, cells) if "no wheel" in cell}
        if said != gap:
            problems.append(
                f"{k2_path}:{number}: the k2 row says no wheel for "
                f"{sorted(said)}, but {INSTALL_K2} skips k2 for {sorted(gap)}"
            )
    # A row that is not there validates nothing, and the loop above would have
    # said so by saying nothing at all - which is the shape of every defect
    # this file was written against.
    if not seen_k2_row and k2_path in text:
        problems.append(
            f"{k2_path}: no row starting {K2_ROW!r}, so nothing records "
            f"that {INSTALL_K2} skips k2 for torch {sorted(gap) or 'nothing'} "
            "and that those tests importorskip rather than fail"
        )
    return problems


def check_versions_are_built_variants() -> list:
    """Every version pinned anywhere must be one image_variants.json builds.

    install_torch.sh exits 1 on a pytorch version outside that set, and the
    python floor is a hard requirement of pyproject.toml, so a version outside
    it is not a slow path - it is a build that cannot succeed. Most of these
    sites are reached only by a human following a README or a scheduled job, so
    nothing turns red when one rots: TH_VERSION sat at 2.7.1 for five months and
    took the weekly docker publish down every Monday.
    """
    pythons, torches = variants()
    allowed = {"python": pythons, "pytorch": torches}
    problems = []
    for name in tracked_files():
        try:
            text = Path(name).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue  # binary, or a symlink into something not checked out

        found = []
        for pattern, axis in LINE_PINS:
            for match in pattern.finditer(text):
                line = text[: match.start()].count("\n") + 1
                found.append((line, match.group(1), axis))
        for key, pattern, axis in MATRIX_PINS:
            for number, line in enumerate(text.split("\n"), start=1):
                if key not in line:
                    continue
                found.extend(
                    (number, version, axis) for version in pattern.findall(line)
                )

        for line, version, axis in sorted(found):
            if version in allowed[axis] or (name, version) in ALLOWED_PINS:
                continue
            problems.append(
                f"{name}:{line}: pins {axis} {version}, which "
                f"ci/image_variants.json does not list "
                f"({', '.join(allowed[axis])}). Move it to a built version, or "
                "add it to ALLOWED_PINS with the reason it is not installed"
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
    # The torch version a branch installs is the one it guards on, not the
    # argument to install_torch - that argument is the torchaudio version, and
    # the two stopped being the same at torch 2.12, where torchaudio's releases
    # end (2.11.0 is its last). Reading the argument, as this did while they
    # matched, reports every torch above 2.11.0 as missing from a script that
    # installs it.
    text = INSTALL_TORCH.read_text()
    installable = set()
    for branch in re.split(r"^(?:el)?if \$\(pytorch_plus ", text, flags=re.M)[1:]:
        guard, _, body = branch.partition("\n")
        if re.search(r"^\s*install_torch \d", body, re.M):
            installable.add(guard.split(")")[0].strip())
    if installable != set(torches):
        problems.append(
            f"{INSTALL_TORCH}: installs {sorted(installable)}, but "
            f"ci/image_variants.json builds {sorted(torches)}. It exits 1 on "
            "anything it does not install"
        )
    return problems


INSTALL_K2 = Path("tools/installers/install_k2.sh")
PREBUILT_ACTION = Path(".github/actions/use-prebuilt-environment/action.yml")
# Not a substring test for the name. A comment mentioning k2_missing_for, or
# the action keeping its own literal copy under that same name, satisfies one
# of those while the import still fires on a torch version with no wheel -
# both were checked against the pre-#6681 action and both passed. What has to
# be there is a read of the installer, and a use of what was read.
READS_GAP_LIST = re.compile(r"k2_missing_for=\$\([^\n]*install_k2\.sh")
USES_GAP_LIST = re.compile(r"\$\{?k2_missing_for\}?")


def check_k2_gap_is_a_built_variant() -> list:
    """install_k2.sh's k2_missing_for may only name torch versions CI builds.

    k2 publishes one wheel per (k2 version, torch version, python version) and
    lags torch by weeks, so a newly supported torch spends a while with no k2 at
    all. install_k2.sh skips itself for those rather than failing every image
    build, and because the k2 tests are pytest.importorskip, that skip is
    invisible in a green run - the list is the only record that one variant
    tests less than the others.

    So the list has to stay tied to the grid. An entry for a torch version
    nobody builds is either a version that was dropped, or one k2 has since
    published and nobody removed; both read as "k2 is handled here" while
    silently keeping it out of the next run that uses that version.
    """
    torches = variants()[1]
    if not INSTALL_K2.exists():
        return [f"{INSTALL_K2}: missing, so its k2 gap list cannot be checked"]
    match = re.search(r'^k2_missing_for="([^"]*)"', INSTALL_K2.read_text(), re.M)
    if match is None:
        return [
            f"{INSTALL_K2}: no k2_missing_for= assignment, so a torch version "
            "k2 has no wheel for fails the whole environment build instead of "
            "skipping k2"
        ]
    problems = [
        f"{INSTALL_K2}: k2_missing_for names torch {version}, which "
        f"ci/image_variants.json does not build ({', '.join(torches)})"
        for version in match.group(1).split()
        if version not in torches
    ]
    # The other half of the same fact, and the half that actually broke.
    # Skipping k2 in the installer is only safe while whatever verifies the
    # environment knows to skip it too: an unconditional `import k2` there
    # fails every job on that torch. #6678 added 2.14.0 to the gap list and
    # left the assertion in .github/actions/use-prebuilt-environment demanding
    # k2, and nothing said so for a day - the runs before the images for that
    # hash were published took the build-from-scratch path, where this action
    # does not run at all. #6681 fixed it; this is what keeps the two from
    # drifting apart again.
    if match.group(1).split():
        if not PREBUILT_ACTION.exists():
            problems.append(f"{PREBUILT_ACTION}: missing")
        else:
            text = PREBUILT_ACTION.read_text()
            gap = " ".join(match.group(1).split())
            why = (
                "  Asserting `import k2` on a torch version k2 publishes no "
                "wheel for fails every job on that part of the grid, and only "
                "once the prebuilt images exist - before that the jobs build "
                "their own environment and never run this action."
            )
            if READS_GAP_LIST.search(text) is None:
                problems.append(
                    f"{PREBUILT_ACTION}: does not read {INSTALL_K2}'s "
                    f"k2_missing_for, which currently skips k2 for torch "
                    f"{gap}.\n"
                    "  Expected an assignment reading the installer, as in "
                    "`k2_missing_for=$(sed ... tools/installers/install_k2.sh)`"
                    ". Naming it in a comment, or keeping a second copy of the "
                    f"list here, is what drifts.\n{why}"
                )
            elif USES_GAP_LIST.search(text) is None:
                problems.append(
                    f"{PREBUILT_ACTION}: reads {INSTALL_K2}'s k2_missing_for "
                    f"and never uses it, so k2 is still asserted on torch "
                    f"{gap}.\n{why}"
                )
    return problems


# Where a failed download takes down a job nobody is watching. Explicit globs
# rather than a walk: tools/ also holds the built venv and a kaldi checkout,
# and the egs2 recipes are deliberately out of scope - a corpus download that
# fails in front of the person who started it gets re-run.
DOWNLOAD_SITES = (
    "ci/*.sh",
    "tools/Makefile",
    "tools/*.sh",
    "tools/installers/*.sh",
    "docker/*.sh",
    "docker/*.dockerfile",
    "docker/prebuilt/*.dockerfile",
    ".devcontainer/*/*.dockerfile",
    ".devcontainer/*/build_image.sh",
    ".github/workflows/*.yml",
    ".github/actions/*/*.yml",
)

# The one wget that is allowed to stand alone, because it *is* the retry loop.
RETRY_HELPER = Path("tools/installers/download_with_retry.sh")

# wget or curl as the command being run rather than as an argument, which is
# what keeps this off "apt-get install -y wget curl bc" and off "choco install
# -y wget". Anything may come between in the form of an assignment prefix
# (FOO=bar wget ...).
COMMAND = re.compile(
    r"""(?: ^ | [|;&(){}] | ! | \b(?:RUN|if|elif|while|until|then|do|else|
                                   sudo|time|exec|eval)\b )
        \s*
        (?: [A-Za-z_][A-Za-z0-9_]* = \S* \s+ )*
        (wget|curl) \b
    """,
    re.X,
)

# A trailing comment is still a comment, so cut the line at the first # that
# starts a word. No URL can contain one - whitespace ends a URL - and ${#var}
# is not preceded by whitespace.
COMMENT = re.compile(r"(?:^|\s)#")

# What curl calls a transient error and therefore retries. wget has to be told
# the same set by hand, so that the two behave alike.
TRANSIENT = ("429", "500", "502", "503", "504")

# A transfer needs something to fetch, which is either a literal URL or an
# expansion holding one. Without this, `wget --version` in the windows
# workflow reads as an unguarded download; with it, a URL kept in a variable
# is still checked - which is the point, since most of these scripts do that.
HAS_TARGET = re.compile(r"https?://|\$\{?\w")
PROBE = re.compile(r"(?:^|\s)(?:--version|--help|-V)(?:\s|$)")

FAIL_FLAG = re.compile(
    r"(?:^|\s)(?:-[A-Za-z]*f[A-Za-z]*|--fail(?:-with-body)?)(?:\s|$)"
)
CURL_RETRY = re.compile(r"--retry[= ](\d+)")
WGET_RETRY = re.compile(r"--retry-on-http-error=(\S+)")

# The nltk CLI, which prompts on failure and never retries.
NLTK_CLI = re.compile(r"python[0-9.]*\s+-m\s+nltk\.downloader")


def logical_lines(text: str) -> list:
    """(line number, code) with comments cut and continuations joined.

    A command split across physical lines is one command, and the URL is
    routinely on the second half - so a scan of physical lines sees a download
    with no options and a bare URL with no command, and reports neither.
    """
    joined, number, parts = [], None, []
    for count, line in enumerate(text.splitlines(), 1):
        comment = COMMENT.search(line)
        code = line[: comment.start()] if comment else line
        number = count if number is None else number
        if code.rstrip().endswith("\\"):
            parts.append(code.rstrip()[:-1])
            continue
        parts.append(code)
        joined.append((number, " ".join(parts)))
        number, parts = None, []
    if parts:
        joined.append((number, " ".join(parts)))
    return joined


def _wget_problem(line: str) -> str:
    """Why this wget would not survive a transient response, or ""."""
    codes = WGET_RETRY.search(line)
    if codes is None:
        return (
            "no --retry-on-http-error, so a 5xx response is fatal on the "
            "first try. --tries does not cover it: against a server "
            "answering 500, wget --tries=3 makes one request and exits 8"
        )
    missing = [code for code in TRANSIENT if code not in codes.group(1).split(",")]
    if missing:
        return (
            f"--retry-on-http-error={codes.group(1)} leaves "
            f"{', '.join(missing)} fatal. Only the codes listed are retried, "
            "so a partial list reads like a retry and is not one"
        )
    return ""


def _curl_problem(line: str) -> str:
    """Why this curl would not survive a transient response, or ""."""
    if FAIL_FLAG.search(line) is None:
        return (
            "no --fail, so curl exits 0 on a 5xx and writes the error body to "
            "the output. Measured: a server answering 500 leaves 'oops' in "
            "the file and curl reports success - which, piped to a shell, "
            "runs it"
        )
    retry = CURL_RETRY.search(line)
    if retry is None:
        return "no --retry, so a transient 5xx is fatal on the first try"
    if int(retry.group(1)) < 1:
        return f"--retry {retry.group(1)} performs no retries at all"
    return ""


def check_downloads_retry_on_5xx() -> list:
    """Every CI download must survive a transient failure."""
    problems = []
    for glob in DOWNLOAD_SITES:
        for path in sorted(Path().glob(glob)):
            if path == RETRY_HELPER:
                continue
            for number, line in logical_lines(path.read_text()):
                if NLTK_CLI.search(line):
                    problems.append(
                        f"{path}:{number}: downloads through the nltk CLI.\n"
                        f"    {line.strip()[:120]}\n"
                        "  `python -m nltk.downloader` calls download() with "
                        "halt_on_error=False, so a failed download prompts "
                        '"Retry? [n/y/e]" and reads stdin - in CI that is an '
                        "EOFError traceback from input() with the real cause "
                        "scrolled off above it, on the first attempt, because "
                        "nltk has no retry. Call "
                        "installers/install_nltk_data.sh instead."
                    )
                    continue
                match = COMMAND.search(line)
                if match is None:
                    continue
                if HAS_TARGET.search(line) is None or PROBE.search(line):
                    continue
                tool = match.group(1)
                why = _wget_problem(line) if tool == "wget" else _curl_problem(line)
                if not why:
                    continue
                problems.append(
                    f"{path}:{number}: {tool} {why}.\n"
                    f"    {line.strip()[:120]}\n"
                    f"  In a shell script, call download_with_retry instead - "
                    f"it retries on a fresh connection, which the wget flags "
                    f"cannot do for a failed TLS handshake, and checks that "
                    f"what came back is what was asked for."
                )
    return problems


LABELER = Path(".github/labeler.yml")
MERGIFY = Path(".mergify.yml")


def _segments(glob: str) -> list:
    """The glob split into path segments, with runs of `*` collapsed.

    Inside a segment `**` means the same as `*` - neither crosses a `/` - so
    collapsing the run changes no answer, and it is what keeps a segment
    pattern from holding two adjacent `.*`.
    """
    return [
        part if part == "**" else re.sub(r"\*+", "*", part) for part in glob.split("/")
    ]


def _after_globstars(states: set, globs: list) -> set:
    """The states reachable by letting `**` match no segments at all."""
    reached, pending = set(states), list(states)
    while pending:
        index = pending.pop()
        if index < len(globs) and globs[index] == "**" and index + 1 not in reached:
            reached.add(index + 1)
            pending.append(index + 1)
    return reached


def _matches(globs: list, path: str) -> bool:
    """Does this glob match this path? The subset labeler.yml uses.

    Segment by segment, tracking which prefixes of the glob are still live,
    rather than as one regular expression. `**` in a regex becomes `.*`, and
    several of those in one pattern backtrack exponentially: the version this
    replaces took 9.6s on a glob with eight `**/` against a 40-segment path,
    and did not finish in 25s with sixteen. This is O(glob segments x path
    segments) with no backtracking between segments.

    It is an approximation of minimatch in one direction only - a trailing
    `**` here also matches zero segments - which cannot matter, because the
    question asked of it is only whether the glob matches any tracked file.
    """
    states = _after_globstars({0}, globs)
    for part in path.split("/"):
        moved = set()
        for index in states:
            if index >= len(globs):
                continue
            if globs[index] == "**":
                moved.add(index)
            elif fnmatch.fnmatchcase(part, globs[index]):
                moved.add(index + 1)
        states = _after_globstars(moved, globs)
        if not states:
            return False
    return len(globs) in states


def check_label_rules() -> list:
    """One mechanism labels by path, and its globs must match something."""
    if not LABELER.exists():
        return [f"{LABELER}: missing"]
    tracked = subprocess.run(
        ["git", "ls-files", "-z"],
        capture_output=True,
        text=True,
        check=False,
    )
    if tracked.returncode != 0:
        return [f"{LABELER}: could not list tracked files: {tracked.stderr.strip()}"]
    paths = [path for path in tracked.stdout.split("\0") if path]
    lines = LABELER.read_text().splitlines()
    problems = []
    for label, rules in yaml.safe_load(LABELER.read_text()).items():
        for rule in rules:
            for kind in rule.get("changed-files", []):
                for glob in kind.get("any-glob-to-any-file", []):
                    globs = _segments(glob)
                    if any(_matches(globs, path) for path in paths):
                        continue
                    number = next(
                        (n for n, ln in enumerate(lines, 1) if glob in ln), None
                    )
                    at = f":{number}" if number else ""
                    problems.append(
                        f"{LABELER}{at}: [{label}] matches no tracked file: {glob}\n"
                        "  A rule that matches nothing applies its label to "
                        "nothing, and nothing else says so. Four rules in "
                        ".mergify.yml were written as globs where the "
                        "condition takes a regex, so ASR, TTS, MT and LM "
                        "matched none of the repository's paths and were "
                        "never applied."
                    )
    problems += _mergify_label_rules()
    return problems


def _mergify_label_rules() -> list:
    """Path-based label rules that came back to .mergify.yml."""
    if not MERGIFY.exists():
        return []
    problems = []
    for rule in yaml.safe_load(MERGIFY.read_text()).get("pull_request_rules", []):
        conditions = [c for c in rule.get("conditions", []) if isinstance(c, str)]
        if not any(c.startswith("files~=") for c in conditions):
            continue
        if "label" not in rule.get("actions", {}):
            continue
        problems.append(
            f"{MERGIFY}: labels by path: {rule.get('name')!r}\n"
            "  Path-based labelling belongs in .github/labeler.yml, which "
            "matches globs. `files~=` here takes a regular expression, and "
            "four rules written as globs meant ASR, TTS, MT and LM were never "
            "applied to anything."
        )
    return problems


def main() -> int:
    bad = (
        check_variants()
        + check_generated_matrices()
        + check_integration_relevance_paths()
        + check_needed_before_read()
        + check_reporter_runs_last()
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
        + check_coverage_tables()
        + check_declared_support_matches_variants()
        + check_k2_gap_is_a_built_variant()
        + check_downloads_retry_on_5xx()
        + check_label_rules()
    )
    for problem in bad:
        print(problem, file=sys.stderr)
    build, consumer = inputs(BUILD), inputs(CONSUMER)
    if build == consumer and not bad:
        print(
            f"hash inputs agree ({len(build)} entries); every job matrix and "
            "every matrix the workflow generates is a "
            "built variant; integration and configuration tasks match their "
            "scripts; every shard set is complete; every test step has "
            "HF_TOKEN; every third-party action is pinned to a SHA; "
            "every checkout drops its credentials; every job declares "
            "permissions; every codecov upload has a token; pyproject declares "
            "no direct references; every python and pytorch version named "
            "outside image_variants.json is one it lists, and what the "
            "package declares matches it; the k2 gap list names only built "
            "variants; every download retries on 5xx; "
            "every labeler glob matches a tracked file; no duplicate keys"
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

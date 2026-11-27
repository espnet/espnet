"""Atlas subcommands for ``espnet3``."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import shutil
import sys
import time
import uuid
from importlib import resources
from pathlib import Path

from filelock import FileLock
from omegaconf import OmegaConf

from espnet3.bin.atlas.agents.cli import CliAgent
from espnet3.bin.atlas.core.config import load_atlas_config
from espnet3.bin.atlas.core.graph import Resources, load_graph
from espnet3.bin.atlas.core.scheduler import AtlasGraphRuntime
from espnet3.bin.atlas.core.state import StateStore
from espnet3.bin.atlas.core.utils import save_yaml


def _configure_logging() -> None:
    """Show ``info``-level Atlas logs on stderr.

    Without this only warnings reach the console, so a controller (login node
    or submitted tick job) looks silent while it is healthy. Only the Atlas
    logger is raised to ``INFO``; other libraries keep their own levels. A
    stderr handler is added only if the root logger has none (importing
    ``espnet3`` usually installs one already).
    """
    if not logging.getLogger().handlers:
        logging.basicConfig(
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
    logging.getLogger("espnet3.bin.atlas").setLevel(logging.INFO)


def _build_agent(config, recipe_dir: Path):
    """Build the coding-agent client configured under ``config.atlas.agent``."""
    return CliAgent(config.atlas.agent, recipe_dir=recipe_dir)


def _resolve_study_dir(recipe_dir: Path, configured: str | Path) -> Path:
    """Resolve a study's configured ``study_dir`` relative to the recipe root."""
    path = Path(configured)
    return path.resolve() if path.is_absolute() else (recipe_dir / path).resolve()


def _load_runtime(study_dir: Path, recipe_dir: Path):
    """Rebuild an ``AtlasGraphRuntime`` from a study dir's saved config/graph/state."""
    config = OmegaConf.load(study_dir / "atlas.yaml")
    OmegaConf.resolve(config)
    graph = load_graph(study_dir / "graph.yaml", recipe_dir=recipe_dir)
    state = StateStore(study_dir)
    trial_resources_default = Resources.from_config(
        getattr(config.atlas.submission, "resources", None)
    )
    return AtlasGraphRuntime(
        study_dir=study_dir,
        recipe_dir=recipe_dir,
        config=config,
        graph=graph,
        state=state,
        trial_resources_default=trial_resources_default,
        agent=_build_agent(config, recipe_dir),
    )


def handle_init(args) -> int:
    """Initialize a study: write its config/graph/program and bootstrap its state.

    Implements ``espnet3 atlas init``. Loads ``args.config`` over the Atlas
    defaults (``resources/atlas_defaults.yaml``), creates
    ``config.atlas.study_dir`` with its ``knowledge``, ``trials`` and
    ``node_runs`` subdirectories, writes the resolved config to
    ``atlas.yaml``, copies the graph named by ``atlas.application`` to
    ``graph.yaml``, copies or creates ``program.md``, and registers the study
    and its first (bootstrap) node run in state. If a node run already exists
    for this study, the state is left as it is and only the written files are
    refreshed.

    Args:
        args: Parsed CLI args with ``recipe_dir`` and ``config`` (path to the
            recipe's ``atlas.yaml``).

    Returns:
        int: Always ``0``.

    Example:
        Running it for a config whose ``study_dir`` is ``exp/atlas/my_study``
        creates this tree (``program.md`` is the file named by
        ``atlas.objective_file``, or a short template if there is none)::

            <recipe_dir>/exp/atlas/my_study/
            |-- atlas.yaml            resolved study config (defaults merged in)
            |-- graph.yaml            copy of the graph named by atlas.application
            |-- program.md            the study's objective, edited by the user
            |-- state_studies.csv     the study record
            |-- state_node_runs.csv   one pending run of the first graph node
            |-- .state.lock           lock file for the state tables
            |-- knowledge/            knowledge pack, written by research_context
            |-- trials/               one directory per trial, filled as trials run
            `-- node_runs/            per-run artifacts of the graph nodes

        Trial records (``state_trials.csv``), job records (``state_jobs.csv``),
        ``state_events.csv`` and ``trials.csv`` appear once a study is run.

        >>> handle_init(argparse.Namespace(recipe_dir=".", config="conf/atlas.yaml"))
        0
    """
    recipe_dir = Path(args.recipe_dir).resolve()
    config_path = Path(args.config).resolve()
    config = load_atlas_config(config_path)
    study_id = str(config.atlas.study_name)

    # 1. The study directory and the subdirectories every study has.
    study_dir = _resolve_study_dir(recipe_dir, config.atlas.study_dir)
    for name in ("knowledge", "trials", "node_runs"):
        (study_dir / name).mkdir(parents=True, exist_ok=True)

    # 2. The graph: a graph YAML in the recipe, else a packaged graph by name.
    application = str(getattr(config.atlas, "application", "") or "")
    graphs_dir = resources.files("espnet3.bin.atlas").joinpath("graphs")
    packaged_graph = graphs_dir.joinpath(f"{application}.yaml")
    if application and (recipe_dir / application).is_file():
        graph_src = (recipe_dir / application).resolve()
    elif application and packaged_graph.is_file():
        with resources.as_file(packaged_graph) as path:
            graph_src = Path(path)
    else:
        names = sorted(
            path.name[: -len(".yaml")]
            for path in graphs_dir.iterdir()
            if path.name.endswith(".yaml")
        )
        raise ValueError(
            f"atlas.application must name a packaged graph ({', '.join(names)}) "
            f"or a graph YAML file in the recipe, got {application!r}"
        )

    # 3. Freeze the study's config and graph inside the study directory. Later
    #    commands (run, status, the trial jobs) read these copies, not the recipe.
    save_yaml(study_dir / "atlas.yaml", config)
    shutil.copyfile(graph_src, study_dir / "graph.yaml")

    # 4. Register the study in the state tables.
    state = StateStore(study_dir)
    state.create_or_load_study(study_id, study_dir)

    # 5. The objective (program.md), unless the study already has one: the file
    #    named by atlas.objective_file, next to the config or in the recipe,
    #    else a short template for the user to fill in.
    program_path = study_dir / "program.md"
    if not program_path.exists():
        objective_file = Path(
            str(getattr(config.atlas, "objective_file", "program.md"))
        )
        sources = (config_path.parent / objective_file, recipe_dir / objective_file)
        source = next((path for path in sources if path.is_file()), None)
        if source is not None:
            shutil.copyfile(source, program_path)
        else:
            program_path.write_text(
                "# Atlas Objective\n\nDescribe the search goal here.\n",
                encoding="utf-8",
            )

    # 6. Queue the first node run, once. Running init again on a study that
    #    already has node runs only refreshes the files above.
    if not state.list_node_runs(study_id):
        runtime = _load_runtime(study_dir, recipe_dir)
        state.create_node_run(
            run_id=f"run_{uuid.uuid4().hex}",
            study_id=study_id,
            node_name=runtime.get_bootstrap_node_name(),
        )
    return 0


def _ensure_tick_job(
    runtime, resources: Resources, exclude_job_id: str | None = None
) -> str | None:
    """Keep exactly one tick job alive: reuse the live one or submit a new one.

    The tick job is the controller running as a scheduler job. It is started by
    ``handle_run`` and replaced by ``_run_controller_worker`` before its walltime
    runs out ("handoff"). Under the controller lock this first corrects the
    recorded status of tick jobs that died, does nothing if a ``STOP`` file
    exists, and submits a new one only if no other tick job is active
    (``exclude_job_id`` is the caller's own job)::

        time ───────────────────────────────►
        trial jobs  [─ trial 1 ─][─ trial 3 ─] ...   (independent)
        tick job A  [tick·tick·tick]··✕      ✕ = walltime
                                   ▲ handoff: A submits B, exits
        tick job B                   [tick·tick·tick]··✕
                                                    ▲ handoff: B submits C, ...
    """
    with FileLock(runtime.study_dir / ".lock_study_controller"):
        # Correct the recorded status of tick jobs that died, so that a dead
        # controller is not mistaken for a live one below.
        runtime.reconcile_tick_jobs(exclude_job_id=exclude_job_id)
        if (runtime.study_dir / "STOP").exists():
            runtime.logger.info("Study STOP file exists; no controller tick submitted.")
            return None
        active_ticks = [
            job
            for job in runtime.state.list_active_jobs(runtime.study_id)
            if job.job_type == "tick" and job.job_id != exclude_job_id
        ]
        if active_ticks:
            runtime.logger.info(
                "Controller already active: %s", active_ticks[0].external_id
            )
            return active_ticks[0].job_id
        return _register_tick_job(runtime, resources)


def _register_tick_job(runtime, resources: Resources):
    """Submit the tick job, which runs ``atlas run --job-id <id>``.

    Its logs and ``job.yaml`` go to ``<study_dir>/ticks/tick_<id>/``. A failed
    submission is written to ``submission_error.txt``, recorded as a
    ``tick_job_submission_failed`` event, and raised again.
    """
    study_dir = runtime.study_dir
    recipe_dir = runtime.recipe_dir
    tick_dir = study_dir / "ticks" / f"tick_{uuid.uuid4().hex}"
    tick_dir.mkdir(parents=True, exist_ok=True)
    job_id = f"ar_tick_{uuid.uuid4().hex}"
    stdout, stderr = tick_dir / "stdout.log", tick_dir / "stderr.log"
    command = [
        sys.executable,
        "-m",
        "espnet3.bin.main",
        "atlas",
        "run",
        "--study-dir",
        str(study_dir),
        "--recipe-dir",
        str(recipe_dir),
        "--job-id",
        job_id,
    ]
    metadata = {
        "job_id": job_id,
        "command": command,
        "resources": resources.__dict__,
        "workdir": str(recipe_dir),
        "job_type": "tick",
    }
    repo_root = Path(__file__).resolve().parents[3]
    pythonpath = os.pathsep.join(
        path for path in (str(repo_root), os.environ.get("PYTHONPATH", "")) if path
    )
    try:
        job = runtime.submit_job(
            job_id=job_id,
            job_type="tick",
            name=f"ar-{study_dir.name.replace('_', '-')}-controller",
            command=command,
            workdir=recipe_dir,
            stdout=stdout,
            stderr=stderr,
            resources=resources,
            env={"PYTHONPATH": pythonpath},
            metadata=metadata,
        )
    except Exception as exc:
        error_path = tick_dir / "submission_error.txt"
        error_path.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        runtime.state.append_event(
            runtime.study_id,
            "tick_job_submission_failed",
            {"job_id": job_id, "error": f"{type(exc).__name__}: {exc}"},
        )
        raise
    tick_metadata = {**metadata, "backend": job.backend, "external_id": job.external_id}
    (tick_dir / "job.yaml").write_text(
        OmegaConf.to_yaml(OmegaConf.create(tick_metadata)), encoding="utf-8"
    )
    return job_id


def handle_run(args) -> int:
    """Run a study's controller: the process that ticks the scheduler.

    Implements ``espnet3 atlas run``. A tick runs the local graph nodes, submits
    trial jobs, notices dead jobs and checks the budget. Trial jobs start the
    next trial themselves when they finish, but only a controller notices a job
    that died. Where it runs is set by ``atlas.submission.controller.run_in``:

    * ``local`` (default): this process is the controller and loops until the
      study is done (run it in tmux on a login node).
    * ``queue``: submit the controller as a tick job through ``_ensure_tick_job``
      and return. The job runs this command with ``--job-id``, which marks the
      process as that tick job, so it loops instead of submitting again.

    Args:
        args: Parsed CLI args with ``study_dir``, ``recipe_dir``,
            ``poll_interval``, and ``job_id`` (set only for a tick job).

    Returns:
        int: Always ``0``, except that unhandled exceptions from the worker
        path propagate after marking the job ``"failed"``.

    Example:
        >>> handle_run(argparse.Namespace(
        ...     study_dir="exp/atlas/s", recipe_dir=".", poll_interval=None,
        ...     job_id=None,
        ... ))
        0
    """
    study_dir = Path(args.study_dir).resolve()
    recipe_dir = Path(args.recipe_dir).resolve()
    _configure_logging()
    config = OmegaConf.load(study_dir / "atlas.yaml")
    submission_cfg = config.atlas.submission
    state = StateStore(study_dir)
    runtime = _load_runtime(study_dir, recipe_dir)
    controller_cfg = getattr(submission_cfg, "controller", None)
    controller_resources = Resources.from_config(
        getattr(controller_cfg, "resources", None)
    )
    job_id = getattr(args, "job_id", None)
    run_in = str(getattr(controller_cfg, "run_in", "local"))
    if run_in == "queue" and not job_id:
        _ensure_tick_job(runtime, controller_resources)
        return 0
    # Only a tick job hands over to a successor; a controller run by hand stays.
    handoff_after = None
    if job_id:
        handoff_after = getattr(controller_cfg, "handoff_after_sec", 0) or None
    started = time.monotonic()
    status = "completed"
    try:
        asyncio.run(
            _run_controller_worker(
                runtime,
                job_id,
                controller_resources,
                handoff_after,
                started,
                args.poll_interval,
            )
        )
    except BaseException:
        status = "failed"
        raise
    finally:
        if job_id:
            state.update_job_status(job_id, status)
    return 0


async def _run_controller_worker(
    runtime, job_id, controller_resources, handoff_after, started, poll_interval=None
):
    """The body of a tick job: tick, log, sleep, repeat.

    Returns when a ``STOP`` file exists, when the study has finished (budget
    spent, nothing pending, no trial job active), or after ``handoff_after``
    seconds, having submitted a successor first (see ``_ensure_tick_job``).
    ``handoff_after=None``, as for a controller started by hand, never hands off.
    The sleep is ``tick_interval_sec_trial_active`` while trials run, else
    ``tick_interval_sec_idle``.
    """
    while True:
        if (runtime.study_dir / "STOP").exists():
            return
        runtime.tick()
        runtime.logger.info(
            "tick done: study=%s running_trials=%d pending_node_runs=%d",
            runtime.study_id,
            len(runtime.state.list_trials(runtime.study_id, status="running")),
            len(runtime.state.get_pending_node_runs(runtime.study_id)),
        )
        if (runtime.study_dir / "STOP").exists():
            return
        if runtime.should_stop():
            pending = runtime.state.get_pending_node_runs(runtime.study_id)
            active = [
                job
                for job in runtime.state.list_active_jobs(runtime.study_id)
                if job.job_id != job_id and job.job_type != "tick"
            ]
            if not pending and not active:
                return
        if handoff_after and time.monotonic() - started >= handoff_after:
            _ensure_tick_job(runtime, controller_resources, exclude_job_id=job_id)
            return
        active_trials = runtime.state.list_trials(runtime.study_id, status="running")
        interval_key = (
            "tick_interval_sec_trial_active"
            if active_trials
            else "tick_interval_sec_idle"
        )
        interval = poll_interval or int(
            getattr(runtime.config.atlas, interval_key, 600 if active_trials else 60)
            or 1
        )
        await asyncio.sleep(max(1, interval))


def handle_status(args) -> int:
    """Print a study's state: trials, best trial, controller activity, and jobs.

    Implements ``espnet3 atlas status``. The first three lines (``study_dir``,
    ``trials``, ``best``) are stable; the rest is a scheduler-independent summary
    read only from the study's recorded state and trial logs. It does not query
    the job scheduler, so a job's recorded status can lag behind reality.

    Args:
        args: Parsed CLI args with ``study_dir``.

    Returns:
        int: Always ``0``.

    Example:
        >>> handle_status(argparse.Namespace(study_dir="exp/atlas/my_study"))
        study_dir: exp/atlas/my_study
        trials: 3
        best: trial_002 score=0.12
        trial_status: running=2 accepted=1
        stop_file: no
        active_jobs: 2
          trial_000003 trial running id=3046983.pbs1
        0
    """
    study_dir = Path(args.study_dir).resolve()
    config = OmegaConf.load(study_dir / "atlas.yaml")
    state = StateStore(study_dir)
    study_id = str(config.atlas.study_name)
    trials = state.list_trials(study_id)
    print(f"study_dir: {study_dir}")
    print(f"trials: {len(trials)}")
    if trials:
        best = state.get_best_trial(
            study_id,
            metric=str(config.atlas.metric.name),
            mode=str(config.atlas.metric.mode),
        )
        if best is not None:
            print(f"best: {best.trial_id} score={best.score}")
    counts: dict[str, int] = {}
    for trial in trials:
        counts[trial.status] = counts.get(trial.status, 0) + 1
    if counts:
        print(
            "trial_status: " + " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        )
    print(f"stop_file: {'yes' if (study_dir / 'STOP').exists() else 'no'}")
    active = state.list_active_jobs(study_id)
    print(f"active_jobs: {len(active)}")
    for job in active:
        print(
            f"  {job.trial_id or '-'} {job.job_type} {job.status} id={job.external_id}"
        )
    return 0


def handle_stop(args) -> int:
    """Request a study stop and mark it ``stopped`` in state.

    Implements ``espnet3 atlas stop``. Writes a ``STOP`` sentinel file the
    running controller/worker loops check on each iteration, so the study
    winds down instead of being killed outright. Updating the study's status
    is best-effort: if ``atlas.yaml``/state cannot be read (for example the
    study was never initialized), the ``STOP`` file is still written and
    this still returns ``0``.

    Args:
        args: Parsed CLI args with ``study_dir``.

    Returns:
        int: Always ``0``.

    Example:
        >>> handle_stop(argparse.Namespace(study_dir="exp/atlas/my_study"))
        0
    """
    study_dir = Path(args.study_dir).resolve()
    (study_dir / "STOP").write_text("stop\n", encoding="utf-8")
    try:
        config = OmegaConf.load(study_dir / "atlas.yaml")
        StateStore(study_dir).update_study_status(
            str(config.atlas.study_name), "stopped"
        )
    except Exception:  # noqa: BLE001
        pass
    return 0


def add_arguments(subparsers) -> None:
    """Register the ``atlas`` subcommand (``init``/``run``/``status``/``stop``).

    Args:
        subparsers: The ``espnet3`` CLI's top-level subparsers action, as
            created by ``parser.add_subparsers()``.

    Returns:
        None.

    Example:
        >>> add_arguments(parser.add_subparsers())
    """
    parser = subparsers.add_parser(
        "atlas",
        help="Run and manage Atlas studies.",
        description="Run and manage ESPnet3 Atlas studies.",
    )
    commands = parser.add_subparsers(dest="atlas_command", required=True)
    init = commands.add_parser("init", help="Initialize a study from a recipe config.")
    init.add_argument("--recipe-dir", required=True)
    init.add_argument("--config", required=True)
    init.set_defaults(func=handle_init)
    run = commands.add_parser(
        "run", help="Run the study controller (in-process or as a tick job)."
    )
    run.add_argument("--study-dir", required=True)
    run.add_argument("--recipe-dir", required=True)
    run.add_argument("--poll-interval", type=float, default=None)
    run.add_argument("--job-id", default=None, help=argparse.SUPPRESS)
    run.set_defaults(func=handle_run)
    status = commands.add_parser("status", help="Show study status and best trial.")
    status.add_argument("--study-dir", required=True)
    status.set_defaults(func=handle_status)
    stop = commands.add_parser("stop", help="Stop a running study.")
    stop.add_argument("--study-dir", required=True)
    stop.set_defaults(func=handle_stop)

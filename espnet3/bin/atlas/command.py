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

from omegaconf import OmegaConf

from espnet3.bin.atlas.agents.cli import CliAgent
from espnet3.bin.atlas.applications.hyperparameter_search.config_schema import (
    load_atlas_config,
    write_resolved_config,
)
from espnet3.bin.atlas.core.graph import Resources, load_graph
from espnet3.bin.atlas.core.scheduler import AtlasGraphRuntime
from espnet3.bin.atlas.core.serialization import file_lock
from espnet3.bin.atlas.core.state import StateStore


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


def _default_graph_path(application: str = "hyperparameter_search") -> Path:
    """Return the packaged default graph YAML for the selected application."""
    if application == "architecture_search":
        package = "espnet3.bin.atlas.applications.architecture_search"
        graph_name = "graph.yaml"
    else:
        package = "espnet3.bin.atlas.applications.hyperparameter_search"
        graph_name = "default_graph.yaml"
    resource = resources.files(package).joinpath(graph_name)
    with resources.as_file(resource) as path:
        return Path(path)


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


def cmd_init(args) -> int:
    """Initialize a study: write its config/graph/program and bootstrap its state.

    Implements ``espnet3 atlas init``. Loads ``args.config`` merged with its
    TEMPLATE defaults, creates ``config.atlas.study_dir``, writes the
    resolved config to ``atlas.yaml``, copies the configured (or packaged
    default) graph to ``graph.yaml``, copies or creates ``program.md``, and
    registers the study and its first (bootstrap) node run in state — unless
    a node run already exists for this study, in which case initialization
    is a no-op beyond refreshing the written files.

    Args:
        args: Parsed CLI args with ``recipe_dir`` and ``config`` (path to the
            recipe's ``atlas.yaml``).

    Returns:
        int: Always ``0``.

    Example:
        >>> cmd_init(argparse.Namespace(recipe_dir=".", config="conf/atlas.yaml"))
        0
    """
    recipe_dir = Path(args.recipe_dir).resolve()
    config_path = Path(args.config).resolve()
    config = load_atlas_config(config_path)
    study_dir = _resolve_study_dir(recipe_dir, config.atlas.study_dir)
    study_dir.mkdir(parents=True, exist_ok=True)
    graph_src = (
        (recipe_dir / str(config.atlas.graph)).resolve()
        if getattr(config.atlas, "graph", None)
        else _default_graph_path(
            str(getattr(config.atlas, "application", "hyperparameter_search"))
        )
    )
    write_resolved_config(study_dir / "atlas.yaml", config)
    shutil.copyfile(graph_src, study_dir / "graph.yaml")
    state = StateStore(study_dir)
    state.create_or_load_study(str(config.atlas.study_name), study_dir)
    program_path = study_dir / "program.md"
    if not program_path.exists():
        objective_file = Path(str(getattr(config.atlas, "objective_file", "program.md")))
        sources = (config_path.parent / objective_file, recipe_dir / objective_file)
        source = next((path for path in sources if path.is_file()), None)
        if source is not None:
            shutil.copyfile(source, program_path)
        else:
            program_path.write_text(
                "# Atlas Objective\n\nDescribe the search goal here.\n",
                encoding="utf-8",
            )
    study_id = str(config.atlas.study_name)
    if not state.list_node_runs(study_id):
        runtime = _load_runtime(study_dir, recipe_dir)
        state.create_node_run(
            run_id=f"run_{uuid.uuid4().hex}",
            study_id=study_id,
            node_name=runtime.bootstrap_node_name(),
        )
    return 0


def _reconcile_tick_jobs(runtime, exclude_job_id: str | None = None) -> None:
    """Repair recorded tick statuses using the submission agent."""
    runtime.reconcile_tick_jobs(exclude_job_id=exclude_job_id)


def _ensure_tick_job(
    runtime, resources: Resources, exclude_job_id: str | None = None
) -> str | None:
    """Reconcile controller state and ensure exactly one tick job is active."""
    with file_lock(runtime.study_dir / ".lock_study_controller"):
        _reconcile_tick_jobs(runtime, exclude_job_id=exclude_job_id)
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
    """Create and submit one controller tick job."""
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
        "--worker",
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
        path
        for path in (str(repo_root), os.environ.get("PYTHONPATH", ""))
        if path
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


def cmd_run(args) -> int:
    """Run a study's controller loop, dispatching to the configured environment.

    Implements ``espnet3 atlas run``. Three modes, chosen by ``args`` and the
    study's ``submission.environment``:

    - ``args.worker`` is set: this process *is* a submitted controller tick
      job; run :func:`_run_controller_worker` until the study stops, a
      handoff interval elapses, or it is interrupted, then record this job's
      final status.
    - ``environment == "local"``: run the scheduler's own
      ``runtime.run_forever`` loop in this process (no job submission).
    - otherwise: submit (or reuse) one controller tick job via
      :func:`_ensure_tick_job` and return immediately.

    Args:
        args: Parsed CLI args with ``study_dir``, ``recipe_dir``,
            ``poll_interval``, ``worker``, and ``job_id``.

    Returns:
        int: Always ``0``, except that unhandled exceptions from the worker
        path propagate after marking the job ``"failed"``.

    Example:
        >>> cmd_run(argparse.Namespace(
        ...     study_dir="exp/atlas/s", recipe_dir=".", poll_interval=None,
        ...     worker=False, job_id=None,
        ... ))
        0
    """
    study_dir = Path(args.study_dir).resolve()
    recipe_dir = Path(args.recipe_dir).resolve()
    _configure_logging()
    config = OmegaConf.load(study_dir / "atlas.yaml")
    submission_cfg = config.atlas.submission
    environment = str(getattr(submission_cfg, "environment", "local"))
    state = StateStore(study_dir)
    study_id = str(config.atlas.study_name)
    runtime = _load_runtime(study_dir, recipe_dir)
    controller_cfg = getattr(submission_cfg, "controller", None)
    controller_resources = Resources.from_config(
        getattr(controller_cfg, "resources", None)
    )
    if args.worker:
        job_id = args.job_id
        handoff_after = int(getattr(controller_cfg, "handoff_after_sec", 0) or 0) or None
        started = time.monotonic()
        worker_status = "completed"
        try:
            asyncio.run(
                _run_controller_worker(
                    runtime, job_id, controller_resources, handoff_after, started
                )
            )
        except BaseException:
            worker_status = "failed"
            raise
        finally:
            if job_id:
                state.update_job_status(job_id, worker_status)
        return 0
    if environment == "local":
        asyncio.run(runtime.run_forever(poll_interval=args.poll_interval))
        return 0
    _ensure_tick_job(runtime, controller_resources)
    return 0


async def _run_controller_worker(runtime, job_id, controller_resources, handoff_after, started):
    """Tick the scheduler in a loop until the study stops or hands off to a new tick job.

    Called from ``cmd_run`` when this process is a submitted controller tick
    job (``args.worker``). Each iteration calls ``runtime.tick()``, then
    returns if a ``STOP`` file appears, if the study has no more pending
    node runs or active (non-tick) jobs, or if ``handoff_after`` seconds have
    elapsed (in which case it submits a replacement tick job before
    returning). Otherwise it sleeps for the idle or trial-active tick
    interval and loops.
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
                job for job in runtime.state.list_active_jobs(runtime.study_id)
                if job.job_id != job_id and job.job_type != "tick"
            ]
            if not pending and not active:
                return
        if handoff_after and time.monotonic() - started >= handoff_after:
            _ensure_tick_job(runtime, controller_resources, exclude_job_id=job_id)
            return
        active_trials = runtime.state.list_trials(runtime.study_id, status="running")
        interval_key = "tick_interval_sec_trial_active" if active_trials else "tick_interval_sec_idle"
        interval = int(getattr(runtime.config.atlas, interval_key, 600 if active_trials else 60) or 1)
        await asyncio.sleep(max(1, interval))


def cmd_status(args) -> int:
    """Print a study's trial count and current best trial.

    Implements ``espnet3 atlas status``.

    Args:
        args: Parsed CLI args with ``study_dir``.

    Returns:
        int: Always ``0``.

    Example:
        >>> cmd_status(argparse.Namespace(study_dir="exp/atlas/my_study"))
        study_dir: exp/atlas/my_study
        trials: 3
        best: trial_002 score=0.12
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
    return 0


def cmd_stop(args) -> int:
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
        >>> cmd_stop(argparse.Namespace(study_dir="exp/atlas/my_study"))
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


def run_from_recipe_config(recipe_dir: Path, config_path: Path, mode: str = "run") -> int:
    """Initialize and run a study from Python, without going through the CLI parser.

    Calls :func:`cmd_init` then :func:`cmd_run` (in local/foreground mode,
    not as a worker) for the study described by ``config_path``. Intended
    for a recipe's own code (for example a ``run.py`` stage) to kick off an
    Atlas study directly.

    Args:
        recipe_dir: Recipe root.
        config_path: Path to the recipe's ``atlas.yaml``.
        mode: Reserved for a future non-``"run"`` mode; currently unused.

    Returns:
        int: The exit code from :func:`cmd_run` (always ``0`` today).

    Example:
        >>> run_from_recipe_config(Path("."), Path("conf/atlas.yaml"))
        0
    """
    recipe_dir, config_path = Path(recipe_dir).resolve(), Path(config_path).resolve()
    cmd_init(argparse.Namespace(recipe_dir=str(recipe_dir), config=str(config_path)))
    config = load_atlas_config(config_path)
    study_dir = _resolve_study_dir(recipe_dir, config.atlas.study_dir)
    return cmd_run(argparse.Namespace(
        study_dir=str(study_dir), recipe_dir=str(recipe_dir), poll_interval=None, worker=False
    ))


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
        "atlas", help="Run and manage Atlas studies.",
        description="Run and manage ESPnet3 Atlas studies.",
    )
    commands = parser.add_subparsers(dest="atlas_command", required=True)
    init = commands.add_parser("init", help="Initialize a study from a recipe config.")
    init.add_argument("--recipe-dir", required=True)
    init.add_argument("--config", required=True)
    init.set_defaults(func=cmd_init)
    run = commands.add_parser("run", help="Run the study using its configured submission environment.")
    run.add_argument("--study-dir", required=True)
    run.add_argument("--recipe-dir", required=True)
    run.add_argument("--poll-interval", type=float, default=None)
    run.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    run.add_argument("--job-id", default=None, help=argparse.SUPPRESS)
    run.set_defaults(func=cmd_run)
    status = commands.add_parser("status", help="Show study status and best trial.")
    status.add_argument("--study-dir", required=True)
    status.set_defaults(func=cmd_status)
    stop = commands.add_parser("stop", help="Stop a running study.")
    stop.add_argument("--study-dir", required=True)
    stop.set_defaults(func=cmd_stop)

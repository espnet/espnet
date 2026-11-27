"""Tests for ``espnet3.bin.atlas.command`` (stop, status, run and tick jobs)."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging

import pytest
from omegaconf import OmegaConf

from espnet3.bin.atlas import command
from espnet3.bin.atlas.core.models import JobRecord, Resources, Trial
from espnet3.bin.atlas.core.state import StateStore

STUDY = "study"


def _write_config(study_dir, extra=None):
    cfg = {
        "atlas": {
            "study_name": STUDY,
            "metric": {"name": "wer", "mode": "min"},
            "submission": {"controller": {}},
            "tick_interval_sec_idle": 3,
            "tick_interval_sec_trial_active": 7,
        }
    }
    if extra:
        cfg["atlas"]["submission"]["controller"].update(extra)
    study_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(cfg), study_dir / "atlas.yaml")


def _trial(trial_id, status, score=None):
    return Trial(
        trial_id=trial_id,
        study_id=STUDY,
        status=status,
        config_patch={},
        resolved_config_path=None,
        rationale="",
        expected_effect="",
        risk="",
        score=score,
        score_name="wer",
        decision=None,
        parent_trial_id=None,
        attempt_count=0,
        created_at="2020-01-01T00:00:00+00:00",
        updated_at="2020-01-01T00:00:00+00:00",
    )


def _job(job_id, job_type="tick", status="running", trial_id=None):
    return JobRecord(
        job_id=job_id,
        study_id=STUDY,
        trial_id=trial_id,
        node_run_id=None,
        backend="pbs",
        external_id=f"ext_{job_id}",
        status=status,
        command_json="{}",
        resources_json=None,
        workdir=None,
        stdout_path=None,
        stderr_path=None,
        job_type=job_type,
        created_at="2020-01-01T00:00:00+00:00",
        updated_at="2020-01-01T00:00:00+00:00",
    )


class _FakeRuntime:
    """Minimal runtime double for the controller loop and tick submission."""

    def __init__(self, study_dir, stop_after=None, done_after=None):
        self.study_dir = study_dir
        self.recipe_dir = study_dir
        self.study_id = STUDY
        self.logger = logging.getLogger("test")
        self.config = OmegaConf.load(study_dir / "atlas.yaml")
        self.state = StateStore(study_dir)
        self.state.create_or_load_study(STUDY, study_dir)
        self.ticks = 0
        self.stop_after = stop_after
        self.done_after = done_after
        self.submitted = []
        self.reconciled = []
        self.fail_submit = False

    def tick(self):
        self.ticks += 1
        if self.stop_after and self.ticks >= self.stop_after:
            (self.study_dir / "STOP").write_text("x")

    def should_stop(self):
        return bool(self.done_after and self.ticks >= self.done_after)

    def reconcile_tick_jobs(self, exclude_job_id=None):
        self.reconciled.append(exclude_job_id)

    def submit_job(self, **kwargs):
        if self.fail_submit:
            raise RuntimeError("qsub down")
        self.submitted.append(kwargs)
        job = _job(kwargs["job_id"])
        self.state.create_job(job)
        return job


@pytest.fixture
def no_sleep(monkeypatch):
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(command.asyncio, "sleep", fake_sleep)
    return sleeps


def test_handle_stop_writes_file_and_marks_study_stopped(tmp_path):
    """Write STOP and set the study status."""
    _write_config(tmp_path)
    StateStore(tmp_path).create_or_load_study(STUDY, tmp_path)

    assert command.handle_stop(argparse.Namespace(study_dir=str(tmp_path))) == 0

    assert (tmp_path / "STOP").read_text() == "stop\n"
    assert StateStore(tmp_path).get_study(STUDY).status == "stopped"


def test_handle_stop_tolerates_uninitialized_study(tmp_path):
    """Still write STOP when atlas.yaml is missing."""
    assert command.handle_stop(argparse.Namespace(study_dir=str(tmp_path))) == 0
    assert (tmp_path / "STOP").is_file()


def test_handle_status_prints_trials_best_and_jobs(tmp_path, capsys):
    """Summarize trials, best score, stop file and active jobs."""
    _write_config(tmp_path)
    state = StateStore(tmp_path)
    state.create_or_load_study(STUDY, tmp_path)
    state.create_trial(_trial("trial_000001", "accepted", 0.2))
    state.create_trial(_trial("trial_000002", "running"))
    state.create_job(_job("j1", job_type="trial", trial_id="trial_000002"))
    state.create_job(_job("j2"))
    (tmp_path / "STOP").write_text("x")

    command.handle_status(argparse.Namespace(study_dir=str(tmp_path)))

    out = capsys.readouterr().out
    assert "trials: 2" in out
    assert "best: trial_000001 score=0.2" in out
    assert "trial_status: accepted=1 running=1" in out
    assert "stop_file: yes" in out and "active_jobs: 2" in out
    assert "trial_000002 trial running id=ext_j1" in out
    assert "  - tick running id=ext_j2" in out


def test_handle_status_without_trials(tmp_path, capsys):
    """Print only the stable lines for an empty study."""
    _write_config(tmp_path)
    StateStore(tmp_path).create_or_load_study(STUDY, tmp_path)

    command.handle_status(argparse.Namespace(study_dir=str(tmp_path)))

    out = capsys.readouterr().out
    assert "trials: 0" in out and "best:" not in out and "stop_file: no" in out


def test_ensure_tick_job_submits_when_none_active(tmp_path):
    """Submit a tick job, write its job.yaml and reconcile first."""
    _write_config(tmp_path)
    runtime = _FakeRuntime(tmp_path)

    job_id = command._ensure_tick_job(runtime, Resources(cpus=2), exclude_job_id="me")

    assert job_id.startswith("ar_tick_") and runtime.reconciled == ["me"]
    (submitted,) = runtime.submitted
    assert submitted["job_type"] == "tick"
    assert submitted["command"][-2:] == ["--job-id", job_id]
    assert "PYTHONPATH" in submitted["env"]
    (job_yaml,) = (tmp_path / "ticks").glob("tick_*/job.yaml")
    assert OmegaConf.load(job_yaml).backend == "pbs"


def test_ensure_tick_job_reuses_active_tick_and_honours_stop(tmp_path):
    """Return the live tick, skip our own job, and do nothing after STOP."""
    _write_config(tmp_path)
    runtime = _FakeRuntime(tmp_path)
    runtime.state.create_job(_job("live"))

    assert command._ensure_tick_job(runtime, Resources()) == "live"
    assert runtime.submitted == []
    assert command._ensure_tick_job(runtime, Resources(), exclude_job_id="live")
    assert len(runtime.submitted) == 1

    (tmp_path / "STOP").write_text("x")
    assert command._ensure_tick_job(runtime, Resources()) is None
    assert len(runtime.submitted) == 1


def test_register_tick_job_records_failed_submission(tmp_path):
    """Write submission_error.txt, log an event and re-raise."""
    _write_config(tmp_path)
    runtime = _FakeRuntime(tmp_path)
    runtime.fail_submit = True

    with pytest.raises(RuntimeError, match="qsub down"):
        command._register_tick_job(runtime, Resources())

    (error,) = (tmp_path / "ticks").glob("tick_*/submission_error.txt")
    assert "RuntimeError: qsub down" in error.read_text()
    events = (tmp_path / "state_events.csv").read_text()
    assert "tick_job_submission_failed" in events


def test_controller_worker_returns_immediately_on_stop_file(tmp_path, no_sleep):
    """Never tick when STOP already exists."""
    _write_config(tmp_path)
    runtime = _FakeRuntime(tmp_path)
    (tmp_path / "STOP").write_text("x")

    asyncio.run(command._run_controller_worker(runtime, None, Resources(), None, 0))

    assert runtime.ticks == 0 and no_sleep == []


def test_controller_worker_sleeps_between_ticks_until_stop(tmp_path, no_sleep):
    """Tick, sleep by idle or active interval, and stop on the STOP file."""
    _write_config(tmp_path)
    runtime = _FakeRuntime(tmp_path, stop_after=3)
    runtime.state.create_trial(_trial("trial_000001", "running"))

    asyncio.run(command._run_controller_worker(runtime, None, Resources(), None, 0))

    assert runtime.ticks == 3
    assert no_sleep == [7, 7]


def test_controller_worker_uses_idle_and_poll_interval(tmp_path, no_sleep):
    """Prefer the poll interval, else the idle interval, with a 1s floor."""
    _write_config(tmp_path)
    runtime = _FakeRuntime(tmp_path, stop_after=3)
    asyncio.run(command._run_controller_worker(runtime, None, Resources(), None, 0))
    assert no_sleep == [3, 3]

    no_sleep.clear()
    (tmp_path / "STOP").unlink()
    runtime = _FakeRuntime(tmp_path, stop_after=2)
    asyncio.run(
        command._run_controller_worker(runtime, None, Resources(), None, 0, 0.2)
    )
    assert no_sleep == [1]


def test_controller_worker_returns_when_study_is_done(tmp_path, no_sleep):
    """Return when should_stop holds with no pending work or active jobs."""
    _write_config(tmp_path)
    runtime = _FakeRuntime(tmp_path, done_after=1)
    runtime.state.create_job(_job("trial_job", job_type="trial"))
    runtime.state.create_node_run("r1", STUDY, "n")

    runtime.stop_after = 4

    asyncio.run(command._run_controller_worker(runtime, None, Resources(), None, 0))
    assert runtime.ticks == 4  # kept ticking while work and jobs were outstanding


def test_controller_worker_finishes_when_nothing_is_left(tmp_path, no_sleep):
    """Return after the first tick when the study is done and idle."""
    _write_config(tmp_path)
    runtime = _FakeRuntime(tmp_path, done_after=1)

    asyncio.run(command._run_controller_worker(runtime, "me", Resources(), None, 0))

    assert runtime.ticks == 1 and no_sleep == []


def test_controller_worker_hands_off_to_a_successor(tmp_path, no_sleep, monkeypatch):
    """Submit a successor and exit once handoff_after has elapsed."""
    _write_config(tmp_path)
    runtime = _FakeRuntime(tmp_path)
    handoffs = []
    monkeypatch.setattr(
        command,
        "_ensure_tick_job",
        lambda rt, res, exclude_job_id=None: handoffs.append(exclude_job_id),
    )
    monkeypatch.setattr(command.time, "monotonic", lambda: 1000.0)

    asyncio.run(command._run_controller_worker(runtime, "me", Resources(), 10, 0.0))

    assert runtime.ticks == 1 and handoffs == ["me"] and no_sleep == []


def _patch_runtime(monkeypatch, runtime):
    monkeypatch.setattr(command, "_load_runtime", lambda *_: runtime)


def test_handle_run_queue_mode_submits_tick_job_and_returns(tmp_path, monkeypatch):
    """Submit the controller as a tick job instead of looping."""
    _write_config(tmp_path, {"run_in": "queue", "resources": {"cpus": 2}})
    runtime = _FakeRuntime(tmp_path)
    _patch_runtime(monkeypatch, runtime)
    seen = []
    monkeypatch.setattr(
        command, "_ensure_tick_job", lambda rt, res: seen.append(res.cpus)
    )

    args = argparse.Namespace(
        study_dir=str(tmp_path), recipe_dir=str(tmp_path), poll_interval=None
    )
    assert command.handle_run(args) == 0
    assert seen == [2] and runtime.ticks == 0


def test_handle_run_local_loops_until_study_done(tmp_path, monkeypatch, no_sleep):
    """Run the controller loop in-process."""
    _write_config(tmp_path)
    runtime = _FakeRuntime(tmp_path, done_after=2)
    _patch_runtime(monkeypatch, runtime)

    args = argparse.Namespace(
        study_dir=str(tmp_path), recipe_dir=str(tmp_path), poll_interval=1.0
    )
    assert command.handle_run(args) == 0
    assert runtime.ticks == 2


def test_handle_run_tick_job_marks_status_completed_or_failed(
    tmp_path, monkeypatch, no_sleep
):
    """Record the tick job's final status, even when the loop raises."""
    _write_config(tmp_path, {"handoff_after_sec": 0})
    runtime = _FakeRuntime(tmp_path, done_after=1)
    runtime.state.create_job(_job("tick_1"))
    _patch_runtime(monkeypatch, runtime)
    args = argparse.Namespace(
        study_dir=str(tmp_path),
        recipe_dir=str(tmp_path),
        poll_interval=None,
        job_id="tick_1",
    )

    command.handle_run(args)
    assert runtime.state.list_jobs(STUDY)[0].status == "completed"

    runtime.state.update_job_status("tick_1", "running")
    runtime.tick = lambda: (_ for _ in ()).throw(ValueError("boom"))
    with pytest.raises(ValueError):
        command.handle_run(args)
    assert runtime.state.list_jobs(STUDY)[0].status == "failed"


def test_configure_logging_sets_atlas_logger_to_info():
    """Raise only the Atlas logger to INFO."""
    command._configure_logging()
    assert logging.getLogger("espnet3.bin.atlas").level == logging.INFO


def test_resolve_study_dir_handles_relative_and_absolute(tmp_path):
    """Resolve relative paths against the recipe directory."""
    assert command._resolve_study_dir(tmp_path, "exp/s") == (tmp_path / "exp/s")
    assert command._resolve_study_dir(tmp_path, tmp_path / "abs") == tmp_path / "abs"


def test_build_agent_creates_cli_agent(tmp_path):
    """Build a CliAgent from the atlas agent config."""
    config = OmegaConf.create({"atlas": {"agent": {"provider": "claude"}}})
    agent = command._build_agent(config, tmp_path)
    assert type(agent).__name__ == "CliAgent"


def _init_args(tmp_path, application, objective=None):
    recipe = tmp_path / "recipe"
    (recipe / "conf").mkdir(parents=True)
    graph = recipe / "graph.yaml"
    graph.write_text(
        "name: g\nversion: 1\nstart: a\nnodes:\n  a:\n"
        "    target: espnet3.bin.atlas.stages.check_budget.CheckBudgetStage\n"
        "    execution: local\n    terminal: true\n",
        encoding="utf-8",
    )
    lines = [
        "atlas:",
        "  study_name: s1",
        "  study_dir: exp/atlas/s1",
        f"  application: {application}",
        "  metric: {name: wer, mode: min}",
        "  submission: {resources: {}}",
        "  agent: {provider: codex}",
    ]
    if objective:
        lines.append(f"  objective_file: {objective}")
        (recipe / "conf" / objective).write_text("my goal", encoding="utf-8")
    config = recipe / "conf" / "atlas.yaml"
    config.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return recipe, argparse.Namespace(recipe_dir=str(recipe), config=str(config))


def test_handle_init_rejects_unknown_application(tmp_path):
    """List packaged graphs when the application cannot be found."""
    recipe, args = _init_args(tmp_path, "nope")
    (recipe / "graph.yaml").unlink()

    with pytest.raises(ValueError, match="atlas.application must name"):
        command.handle_init(args)


def test_handle_init_copies_objective_and_is_idempotent(tmp_path, monkeypatch):
    """Create study files once and keep node runs on a second init."""
    recipe, args = _init_args(tmp_path, "graph.yaml", objective="goal.md")
    monkeypatch.setattr(command, "_build_agent", lambda *_: object())

    assert command.handle_init(args) == 0
    study = recipe / "exp" / "atlas" / "s1"
    assert (study / "program.md").read_text() == "my goal"
    runs = StateStore(study).list_node_runs("s1")
    assert [r["node_name"] for r in runs] == ["a"]

    command.handle_init(args)
    assert len(StateStore(study).list_node_runs("s1")) == 1


def test_handle_init_uses_packaged_graph_and_template_program(tmp_path, monkeypatch):
    """Fall back to a packaged graph and a template objective."""
    recipe, args = _init_args(tmp_path, "hyperparameter_search")
    (recipe / "graph.yaml").unlink()
    monkeypatch.setattr(command, "_build_agent", lambda *_: object())

    command.handle_init(args)

    study = recipe / "exp" / "atlas" / "s1"
    assert "Atlas Objective" in (study / "program.md").read_text()
    assert json.dumps(
        OmegaConf.to_container(OmegaConf.load(study / "graph.yaml"))
    ).count("propose_trial")


def test_add_arguments_registers_atlas_subcommands():
    """Parse each atlas subcommand to its handler."""
    parser = argparse.ArgumentParser()
    command.add_arguments(parser.add_subparsers())

    run = parser.parse_args(
        ["atlas", "run", "--study-dir", "s", "--recipe-dir", "r", "--job-id", "j"]
    )
    assert run.func is command.handle_run and run.job_id == "j"
    stop = parser.parse_args(["atlas", "stop", "--study-dir", "s"])
    assert stop.func is command.handle_stop
    status = parser.parse_args(["atlas", "status", "--study-dir", "s"])
    assert status.func is command.handle_status

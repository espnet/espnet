"""Tests for espnet3.bin.atlas.stages.run_trial."""

from __future__ import annotations

import logging
import subprocess
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from espnet3.bin.atlas.core.models import AgentResponse, StageContext
from espnet3.bin.atlas.core.state import StateStore
from espnet3.bin.atlas.stages import debug_trial, propose_trial, run_trial
from espnet3.bin.atlas.stages.propose_trial import ProposeTrialStage
from espnet3.bin.atlas.stages.run_trial import RunTrialStage
from espnet3.bin.atlas.stages.stage_status import _read_stage_status, _set_stage_status

TRAINING_YAML = """\
optimizer:
  lr: 0.01
  eps: 1.0e-8
tokenizer:
  vocab_size: 100
model:
  vocab_size: ${tokenizer.vocab_size}
dataset:
  preprocessor:
    speed: 1
trainer:
  max_epochs: 2
  logger:
    name: base
exp_dir: base_exp
"""
INFERENCE_YAML = "inference_dir: base_inf\nbeam: 4\nexp_dir: x\n"
METRICS_YAML = "metric: wer\ninference_dir: base_inf\n"


def _fake_load(path, config_name, resolve=True, **_kwargs):
    """Load a recipe-relative YAML without merging packaged defaults."""
    return OmegaConf.load(path)


class _FakeAgent:
    """Return queued responses and record every request."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def run(self, request, artifact_dir=None):
        self.requests.append((request, artifact_dir))
        return self.responses.pop(0)


class _FakeScheduler:
    """Stand in for the graph runtime with the few hooks the stages use."""

    def __init__(self, stop=False, on_check=None, interval=0.0):
        self.study_id = "study"
        self.stop = stop
        self.on_check = on_check
        self.interval = interval
        self.checks = []

    def should_stop(self):
        return self.stop

    def get_gpu_monitor_interval_sec(self):
        return self.interval

    def _maybe_check_keypoints(self, trial_id):
        self.checks.append(trial_id)
        if self.on_check:
            self.on_check(trial_id)


def _proposal(patch=None, **extra):
    """Build a schema-shaped successful propose reply."""
    pairs = [{"key": k, "value": v} for k, v in (patch or {}).items()]
    structured = {
        "rationale": "try it",
        "config_patches": {"training": pairs},
        "expected_effect": "better",
        "risk": "low",
        "extra_stages": [],
    }
    structured.update(extra)
    return AgentResponse(
        status="success",
        message="ok",
        content="",
        structured=structured,
        thread_id="thread-1",
    )


def _debug_reply(training=(), inference=(), **extra):
    """Build a schema-shaped successful debug reply."""
    structured = {
        "root_cause": "bad lr",
        "fix_strategy": "lower lr",
        "files_modified": [],
        "config_patches": {
            "training": [{"key": k, "value": v} for k, v in dict(training).items()],
            "inference": [{"key": k, "value": v} for k, v in dict(inference).items()],
        },
    }
    structured.update(extra)
    return AgentResponse(
        status="success", message="ok", content="", structured=structured
    )


def _failure():
    return AgentResponse(status="failure", message="boom", content="")


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Build a tiny recipe, state store and context factory."""
    recipe = tmp_path / "egs3" / "corpus" / "esp2_asr"
    (recipe / "conf").mkdir(parents=True)
    (recipe / "conf" / "training.yaml").write_text(TRAINING_YAML)
    (recipe / "conf" / "inference.yaml").write_text(INFERENCE_YAML)
    (recipe / "conf" / "metrics.yaml").write_text(METRICS_YAML)
    study_dir = tmp_path / "study"
    (study_dir / "knowledge").mkdir(parents=True)
    (study_dir / "program.md").write_text("Minimize WER.")
    state = StateStore(study_dir)
    state.create_or_load_study("study", study_dir)
    for module in (propose_trial, debug_trial):
        monkeypatch.setattr(module, "load_and_merge_config", _fake_load)

    def make(
        agent=None,
        scheduler=None,
        trial=None,
        atlas=None,
        gpu=False,
        attempt_id="attempt_000001",
    ):
        atlas_cfg = {
            "application": "hyperparameter_search",
            "metric": {"name": "wer"},
            "search_space": {},
            "recipe": {
                "training_config": "conf/training.yaml",
                "inference_config": "conf/inference.yaml",
                "metrics_config": "conf/metrics.yaml",
            },
            "trial": {"max_attempts": 3, "timeout_sec": 10},
        }
        for key, value in (atlas or {}).items():
            if isinstance(value, dict) and isinstance(atlas_cfg.get(key), dict):
                atlas_cfg[key].update(value)
            else:
                atlas_cfg[key] = value
        return StageContext(
            study_dir=study_dir,
            recipe_dir=recipe,
            node_name="node",
            run_id="run_1",
            attempt_id=attempt_id,
            graph=None,
            config=OmegaConf.create({"atlas": atlas_cfg}),
            state=state,
            agent=agent,
            logger=logging.getLogger("test_trials"),
            scheduler=scheduler or _FakeScheduler(),
            current_trial=trial,
            gpu_self_controller=gpu,
        )

    return SimpleNamespace(
        make=make, state=state, study_dir=study_dir, recipe=recipe, tmp=tmp_path
    )


def _propose(env, patch=None, **kwargs):
    """Run propose once and return the result, context, and agent."""
    agent = _FakeAgent(_proposal(patch or {"optimizer.lr": 0.001}))
    context = env.make(agent=agent, **kwargs)
    return ProposeTrialStage().run(context), context, agent


def _load_yaml(path):
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


# ------------------------------------------------------------------- run_trial


class _FakeRun:
    """Replacement for ``subprocess.run`` recording commands by stage."""

    def __init__(self, codes=None, raise_on=None):
        self.codes = codes or {}
        self.raise_on = raise_on
        self.calls = []

    def __call__(self, command, **kwargs):
        stage = RunTrialStage._get_command_stage(command)
        self.calls.append((stage, command, kwargs))
        if self.raise_on == stage:
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return SimpleNamespace(returncode=self.codes.get(stage, 0))


def _trial_context(env, monkeypatch, runner=None, **kwargs):
    """Create a proposed trial and a context ready for ``RunTrialStage``."""
    _propose(env, {"optimizer.lr": 0.001})
    trial = env.state.get_trial("trial_000001")
    runner = runner or _FakeRun()
    monkeypatch.setattr(run_trial.subprocess, "run", runner)
    return env.make(trial=trial, **kwargs), runner


def test_run_trial_runs_train_infer_measure_and_marks_completion(env, monkeypatch):
    """Run the default pipeline, record statuses and collect metrics."""
    context, runner = _trial_context(env, monkeypatch)
    trial_dir = env.study_dir / "trials" / "trial_000001"
    (trial_dir / "inference").mkdir()
    (trial_dir / "inference" / "metrics.json").write_text('{"wer": 1}')
    csv_dir = trial_dir / "exp" / "csv_logs" / "v0"
    csv_dir.mkdir(parents=True)
    (csv_dir / "metrics.csv").write_text("step,loss\n")

    result = RunTrialStage().run(context)

    assert result.status == "success"
    assert [call[0] for call in runner.calls] == ["train", "infer", "measure"]
    train_cmd = runner.calls[0][1]
    assert train_cmd[1:3] == ["run.py", "--stages"]
    assert "--inference_config" not in train_cmd
    assert "--metrics_config" in runner.calls[2][1]
    assert runner.calls[0][2]["timeout"] == 10
    stages = _read_stage_status(trial_dir)["stages"]
    assert {s["status"] for s in stages.values()} == {"completed"}
    trial = env.state.get_trial("trial_000001")
    assert trial.status == "completed"
    assert trial.attempt_count == 1
    assert (trial_dir / "metrics.json").read_text() == '{"wer": 1}'
    assert (trial_dir / "training_metrics.csv").is_file()
    script = (trial_dir / "train_eval.sh").read_text()
    assert script.startswith("#!/usr/bin/env bash")
    assert "--stages train" in script
    job = _load_yaml(trial_dir / "attempts" / "attempt_000001" / "job.yaml")
    assert len(job["commands"]) == 3


def test_run_trial_reports_failure_and_marks_the_stage_failed(env, monkeypatch):
    """Stop at the first non-zero exit and record the failing stage."""
    runner = _FakeRun(codes={"infer": 3})
    context, _ = _trial_context(env, monkeypatch, runner)

    result = RunTrialStage().run(context)

    assert result.status == "failure"
    assert "Stage 'infer' failed with exit code 3" in result.message
    assert [call[0] for call in runner.calls] == ["train", "infer"]
    trial_dir = env.study_dir / "trials" / "trial_000001"
    stages = _read_stage_status(trial_dir)["stages"]
    assert stages["infer"]["status"] == "failed"
    assert stages["infer"]["return_code"] == 3
    assert env.state.get_trial("trial_000001").status == "failed"


def test_run_trial_reports_timeout_when_a_command_times_out(env, monkeypatch):
    """Map ``TimeoutExpired`` to a timeout result and stage status."""
    runner = _FakeRun(raise_on="train")
    context, _ = _trial_context(env, monkeypatch, runner)

    result = RunTrialStage().run(context)

    assert result.status == "timeout"
    trial_dir = env.study_dir / "trials" / "trial_000001"
    assert _read_stage_status(trial_dir)["stages"]["train"]["status"] == "timeout"
    assert env.state.get_trial("trial_000001").status == "timeout"


def test_run_trial_skips_completed_stages_and_resumes_from_checkpoint(env, monkeypatch):
    """Skip finished stages, and resume training from ``exp/last.ckpt``."""
    context, runner = _trial_context(env, monkeypatch)
    trial_dir = env.study_dir / "trials" / "trial_000001"
    _ensure = _set_stage_status
    _ensure(trial_dir, "infer", "completed", "a0")
    _ensure(trial_dir, "measure", "completed", "a0")
    (trial_dir / "exp").mkdir(exist_ok=True)
    (trial_dir / "exp" / "last.ckpt").write_text("ckpt")

    result = RunTrialStage().run(context)

    assert result.status == "success"
    assert [call[0] for call in runner.calls] == ["train"]
    resume_cfg = (
        trial_dir / "attempts" / "attempt_000001" / ("resolved_training_config.yaml")
    )
    assert str(resume_cfg) in runner.calls[0][1]
    assert "ckpt_path" in resume_cfg.read_text()


def test_run_trial_clears_stale_inference_locks_before_retrying_infer(env, monkeypatch):
    """Remove leftover shard locks when infer previously failed."""
    context, runner = _trial_context(env, monkeypatch)
    trial_dir = env.study_dir / "trials" / "trial_000001"
    _set_stage_status(trial_dir, "train", "completed", "a0")
    _set_stage_status(trial_dir, "infer", "failed", "a0")
    lock = trial_dir / "inference" / "dev" / "split.1" / "lock"
    lock.parent.mkdir(parents=True)
    lock.write_text("")

    RunTrialStage().run(context)

    assert not lock.exists()
    assert [call[0] for call in runner.calls] == ["infer", "measure"]


def test_run_trial_uses_the_configured_command_template(env, monkeypatch):
    """Run a single formatted custom command and log it to the train log."""
    atlas = {"trial": {"command_template": ["echo", "{trial_dir}", "{recipe_dir}"]}}
    context, runner = _trial_context(env, monkeypatch, atlas=atlas)

    result = RunTrialStage().run(context)

    assert result.status == "success"
    ((stage, command, _),) = runner.calls
    trial_dir = env.study_dir / "trials" / "trial_000001"
    assert stage == "custom_command"
    assert command == ["echo", str(trial_dir), str(env.recipe)]


def test_run_trial_adds_extra_stages_but_not_graph_only_ones(env, monkeypatch):
    """Run recipe extra stages before train, skipping graph-only stages."""
    context, runner = _trial_context(env, monkeypatch)
    context.current_trial.extra_stages = ["implement_code", "train_tokenizer"]

    RunTrialStage().run(context)

    assert [call[0] for call in runner.calls] == [
        "train_tokenizer",
        "train",
        "infer",
        "measure",
    ]


def test_run_trial_helpers_classify_commands(tmp_path):
    """Classify stage names, training commands and resume configs."""
    stage = RunTrialStage()
    assert stage._is_training_command(["x", "--stages", "train"])
    assert not stage._is_training_command(["x", "--stages", "infer"])
    assert stage._is_training_command(["custom"])
    assert RunTrialStage._get_command_stage(["custom"]) == "custom_command"
    assert RunTrialStage._is_training_stage("collect_stats")
    assert not RunTrialStage._is_training_stage("infer")
    assert (
        RunTrialStage._prepare_resume_config(
            tmp_path / "t.yaml", tmp_path, tmp_path / "missing.ckpt"
        )
        is None
    )
    stage._copy_training_curve_csv(tmp_path, tmp_path)
    assert not (tmp_path / "training_metrics.csv").exists()


class _FakePopen:
    """Popen stand-in that keeps running for a number of polls."""

    instances = []

    def __init__(self, command, polls=3, code=0, **kwargs):
        self.command = command
        self.polls = polls
        self.returncode = code
        self.terminated = False
        self.killed = False
        self.kwargs = kwargs
        _FakePopen.instances.append(self)

    def poll(self):
        if self.terminated or self.killed:
            return self.returncode
        if self.polls <= 0:
            return self.returncode
        self.polls -= 1
        return None

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        return self.returncode


def _patch_popen(monkeypatch, **popen_kwargs):
    _FakePopen.instances = []
    monkeypatch.setattr(
        run_trial.subprocess,
        "Popen",
        lambda command, **kw: _FakePopen(command, **popen_kwargs, **kw),
    )
    monkeypatch.setattr(run_trial.time, "sleep", lambda _s: None)


def test_run_trial_gpu_monitor_completes_training_with_keypoint_checks(
    env, monkeypatch
):
    """Run training under the monitor and check keypoints periodically."""
    _patch_popen(monkeypatch)
    scheduler = _FakeScheduler(interval=0.0)
    context, runner = _trial_context(env, monkeypatch, gpu=True, scheduler=scheduler)

    result = RunTrialStage().run(context)

    assert result.status == "success"
    assert scheduler.checks
    assert [call[0] for call in runner.calls] == ["infer", "measure"]
    assert len(_FakePopen.instances) == 1


def test_run_trial_gpu_monitor_reports_nonzero_training_exit(env, monkeypatch):
    """Fail the trial when monitored training exits non-zero."""
    _patch_popen(monkeypatch, polls=0, code=2)
    context, runner = _trial_context(env, monkeypatch, gpu=True)

    result = RunTrialStage().run(context)

    assert result.status == "failure"
    assert "Stage 'train' failed with exit code 2" in result.message
    assert runner.calls == []


def test_run_trial_gpu_monitor_stops_training_when_early_stopped(env, monkeypatch):
    """Terminate training once the keypoint check marks the trial early stopped."""
    _patch_popen(monkeypatch, polls=100)

    def stop(trial_id):
        trial = env.state.get_trial(trial_id)
        trial.status = "early_stopped"
        env.state.update_trial(trial)

    scheduler = _FakeScheduler(on_check=stop)
    context, runner = _trial_context(env, monkeypatch, gpu=True, scheduler=scheduler)

    result = RunTrialStage().run(context)

    assert result.status == "early_stopped"
    assert _FakePopen.instances[0].terminated
    trial_dir = env.study_dir / "trials" / "trial_000001"
    stages = _read_stage_status(trial_dir)["stages"]
    assert stages["train"]["status"] == "early_stopped"
    assert runner.calls == []


def test_run_trial_gpu_monitor_survives_a_failing_keypoint_check(env, monkeypatch):
    """Log and continue when the in-job early-stop check raises."""
    _patch_popen(monkeypatch, polls=2)

    def boom(_trial_id):
        raise RuntimeError("agent down")

    scheduler = _FakeScheduler(on_check=boom)
    context, _ = _trial_context(env, monkeypatch, gpu=True, scheduler=scheduler)

    result = RunTrialStage().run(context)

    assert result.status == "success"
    assert len(scheduler.checks) >= 1


def test_run_trial_gpu_monitor_times_out_a_long_training(env, monkeypatch):
    """Terminate the training process once the timeout elapses."""
    _patch_popen(monkeypatch, polls=100)
    context, _ = _trial_context(
        env,
        monkeypatch,
        gpu=True,
        scheduler=_FakeScheduler(interval=1000.0),
        atlas={"trial": {"timeout_sec": 0}},
    )

    result = RunTrialStage().run(context)

    assert result.status == "timeout"
    assert _FakePopen.instances[0].terminated
    assert env.state.get_trial("trial_000001").status == "timeout"
    trial_dir = env.study_dir / "trials" / "trial_000001"
    assert _read_stage_status(trial_dir)["stages"]["train"]["status"] == "timeout"


def test_run_trial_gpu_monitor_kills_a_process_that_ignores_terminate(env, monkeypatch):
    """Kill the process if it does not exit after terminate on early stop."""
    _patch_popen(monkeypatch, polls=100)

    def stop(trial_id):
        trial = env.state.get_trial(trial_id)
        trial.status = "early_stopped"
        env.state.update_trial(trial)

    def slow_wait(self, timeout=None):
        if not self.killed:
            raise subprocess.TimeoutExpired("x", timeout)
        return 0

    monkeypatch.setattr(_FakePopen, "wait", slow_wait)
    context, _ = _trial_context(
        env, monkeypatch, gpu=True, scheduler=_FakeScheduler(on_check=stop)
    )

    result = RunTrialStage().run(context)

    assert result.status == "early_stopped"
    assert _FakePopen.instances[0].killed


def test_run_trial_stage_execute_writes_failure_marker(env, monkeypatch):
    """Cache a failure result under the node run's artifact directory."""
    runner = _FakeRun(codes={"train": 1})
    context, _ = _trial_context(env, monkeypatch, runner)

    result = RunTrialStage().execute(context)

    assert result.status == "failure"
    assert (context.get_artifact_dir() / ".stage_failure.json").is_file()

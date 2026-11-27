"""Tests for espnet3.bin.atlas.stages.debug_trial."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from espnet3.bin.atlas.core.models import AgentResponse, StageContext
from espnet3.bin.atlas.core.state import StateStore
from espnet3.bin.atlas.stages import debug_trial, propose_trial
from espnet3.bin.atlas.stages.debug_trial import DebugTrialStage
from espnet3.bin.atlas.stages.propose_trial import ProposeTrialStage
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


# ----------------------------------------------------------------- debug_trial


def _proposed_context(env, agent, **kwargs):
    """Propose a trial and return a context positioned on it for debugging."""
    _propose(env, {"optimizer.lr": 0.001})
    trial = env.state.get_trial("trial_000001")
    return env.make(agent=agent, trial=trial, **kwargs)


def test_debug_trial_retries_unchanged_when_the_agent_fails(env):
    """Retry the same trial without a fix if the debug agent errors."""
    context = _proposed_context(env, _FakeAgent(_failure()))

    result = DebugTrialStage().run(context)

    assert result.status == "continue"
    assert result.next_node == "run_trial"


def test_debug_trial_reflects_when_the_agent_fails_at_the_attempt_limit(env):
    """Route to reflect when the agent fails and no attempts remain."""
    context = _proposed_context(env, _FakeAgent(_failure()))
    context.current_trial.attempt_count = 3

    result = DebugTrialStage().run(context)

    assert result.next_node == "reflect"
    assert "attempt limit" in result.message


def test_debug_trial_reflects_after_a_successful_reply_at_the_attempt_limit(env):
    """Record the debug note and reflect when attempts are exhausted."""
    context = _proposed_context(
        env, _FakeAgent(_debug_reply(training={"optimizer.lr": 0.0001}))
    )
    context.current_trial.attempt_count = 3

    result = DebugTrialStage().run(context)

    assert result.next_node == "reflect"
    assert "root_cause: bad lr" in env.state.get_trial("trial_000001").reflection


def test_debug_trial_retries_unchanged_when_there_is_no_fix(env):
    """Retry the same trial when the agent declines to change anything."""
    context = _proposed_context(env, _FakeAgent(_debug_reply(should_retry=False)))

    result = DebugTrialStage().run(context)

    assert result.status == "success"
    assert result.next_node == "run_trial"
    trial_dir = env.study_dir / "trials" / "trial_000001"
    assert "should_retry: False" in (trial_dir / "debug.md").read_text()
    assert "bad lr" in env.state.get_trial("trial_000001").reflection


def test_debug_trial_applies_a_training_patch_and_invalidates_stages(env):
    """Re-resolve configs, merge the patch and mark affected stages pending."""
    trial_dir = env.study_dir / "trials" / "trial_000001"
    reply = _debug_reply(
        training={"optimizer.lr": 0.0005, "dataset.preprocessor.speed": 3}
    )
    context = _proposed_context(env, _FakeAgent(reply))
    for stage in ("collect_stats", "train", "infer", "measure"):
        _set_stage_status(trial_dir, stage, "completed", "a1")

    result = DebugTrialStage().run(context)

    assert result.status == "success"
    assert result.next_node == "run_trial"
    training = _load_yaml(trial_dir / "resolved_training_config.yaml")
    assert training["optimizer"]["lr"] == 0.0005
    assert training["dataset"]["preprocessor"]["speed"] == 3
    patch = env.state.get_trial("trial_000001").config_patch
    assert patch == {"optimizer.lr": 0.0005, "dataset.preprocessor.speed": 3}
    assert (trial_dir / "resolved_metrics_config.yaml").is_file()
    stages = _read_stage_status(trial_dir)["stages"]
    assert all(
        stages[s]["status"] == "pending"
        for s in ("collect_stats", "train", "infer", "measure")
    )
    assert stages["train"]["history"][-1]["reason"] == "debug config patch applied"


def test_debug_trial_applies_an_inference_patch_and_strips_runtime_keys(env):
    """Patch inference config, drop path-redirecting keys, keep the patch file."""
    trial_dir = env.study_dir / "trials" / "trial_000001"
    reply = _debug_reply(inference={"beam": 8, "inference_dir": "/evil", "": 1})
    context = _proposed_context(env, _FakeAgent(reply))
    for stage in ("train", "infer", "measure"):
        _set_stage_status(trial_dir, stage, "completed", "a1")
    (trial_dir / "inference_config_patch.yaml").write_text("old: 1\n")

    result = DebugTrialStage().run(context)

    assert result.next_node == "run_trial"
    inference = _load_yaml(trial_dir / "resolved_inference_config.yaml")
    assert inference["beam"] == 8
    assert inference["inference_dir"] == str(trial_dir / "inference")
    assert _load_yaml(trial_dir / "inference_config_patch.yaml") == {
        "old": 1,
        "beam": 8,
    }
    stages = _read_stage_status(trial_dir)["stages"]
    assert stages["train"]["status"] == "completed"
    assert stages["infer"]["status"] == "pending"
    assert stages["measure"]["status"] == "pending"
    assert (
        "removed_inference_keys: ['inference_dir', '']"
        in (trial_dir / "debug.md").read_text()
    )


def test_debug_trial_invalidates_every_stage_after_a_code_fix(env):
    """Treat a source edit as invalidating all recorded stages."""
    trial_dir = env.study_dir / "trials" / "trial_000001"
    reply = _debug_reply(files_modified=["src/model.py"])
    context = _proposed_context(env, _FakeAgent(reply))
    for stage in ("train", "infer"):
        _set_stage_status(trial_dir, stage, "completed", "a1")

    result = DebugTrialStage().run(context)

    assert result.status == "success"
    assert result.next_node is None
    stages = _read_stage_status(trial_dir)["stages"]
    assert {s["status"] for s in stages.values()} == {"pending"}


def test_debug_trial_request_includes_attempt_logs_and_respects_code_edit(env):
    """Expose the latest attempt's logs and honour ``allow_pixi=False``."""
    agent = _FakeAgent(_failure())
    context = _proposed_context(
        env,
        agent,
        atlas={"code_edit": {"allowed_dirs": ["src"], "allow_pixi": False}},
    )
    context.current_trial.attempt_count = 2
    (env.study_dir / "trials.csv").write_text("h\n")
    (env.study_dir / "knowledge" / "knowledge_pack.md").write_text("KP")
    trial_dir = env.study_dir / "trials" / "trial_000001"
    (trial_dir / "inference_config_patch.yaml").write_text("")

    DebugTrialStage().run(context)

    request, _ = agent.requests[0]
    assert request.task == "debug"
    assert request.context_files["attempt_stdout"].endswith(
        "attempts/attempt_000002/stdout.log"
    )
    assert request.allowed_actions == ["file_write"]
    assert request.knowledge == "KP"
    assert request.trial_history_csv == "h\n"


def test_debug_trial_filters_training_patch_by_search_space(env):
    """Drop denied training keys; a fully denied patch means no fix."""
    atlas = {"search_space": {"denied_keys": ["optimizer.*"]}}
    context = _proposed_context(
        env,
        _FakeAgent(_debug_reply(training={"optimizer.lr": 9.0})),
        atlas=atlas,
    )

    result = DebugTrialStage().run(context)

    assert result.next_node == "run_trial"
    assert "Debug found no fix" in result.message
    assert env.state.get_trial("trial_000001").config_patch["optimizer.lr"] == 0.001

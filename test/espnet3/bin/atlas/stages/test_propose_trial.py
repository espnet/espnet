"""Tests for propose_trial."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from espnet3.bin.atlas.core.models import AgentResponse, StageContext
from espnet3.bin.atlas.core.state import StateStore
from espnet3.bin.atlas.stages import debug_trial, propose_trial
from espnet3.bin.atlas.stages.propose_trial import ProposeTrialStage

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


def test_propose_trial_writes_resolved_configs_for_a_schema_reply(env):
    """Turn a schema-shaped reply into a proposed trial with configs on disk."""
    result, context, agent = _propose(env)

    assert result.status == "success"
    assert result.payload == {"trial_id": "trial_000001"}
    trial_dir = env.study_dir / "trials" / "trial_000001"
    training = _load_yaml(trial_dir / "resolved_training_config.yaml")
    assert training["optimizer"]["lr"] == 0.001
    assert training["model"]["vocab_size"] == 100
    assert training["exp_tag"] == "trial_000001"
    assert training["exp_dir"] == str(trial_dir / "exp")
    inference = _load_yaml(trial_dir / "resolved_inference_config.yaml")
    assert inference["inference_dir"] == str(trial_dir / "inference")
    metrics = _load_yaml(trial_dir / "resolved_metrics_config.yaml")
    assert metrics["metric"] == "wer"
    assert _load_yaml(trial_dir / "config_patch.yaml") == {"optimizer.lr": 0.001}
    assert (trial_dir / "rationale.md").read_text() == "try it"
    stored = env.state.get_trial("trial_000001")
    assert stored.status == "proposed"
    assert stored.codex_thread_id == "thread-1"
    assert stored.config_patch["optimizer.lr"] == 0.001
    assert stored.resolved_config_path == trial_dir / "resolved_training_config.yaml"
    assert result.artifacts["training_config"].endswith("resolved_training_config.yaml")
    request, artifact_dir = agent.requests[0]
    assert request.task == "propose"
    assert request.objective == "Minimize WER."
    assert artifact_dir == trial_dir
    assert request.allowed_actions == ["config_patch_only"]
    assert "lr: 0.01" in request.repo_context


def test_propose_trial_returns_no_budget_when_scheduler_stops(env):
    """Short-circuit without calling the agent when the budget is spent."""
    agent = _FakeAgent()
    context = env.make(agent=agent, scheduler=_FakeScheduler(stop=True))

    result = ProposeTrialStage().run(context)

    assert result.status == "no_budget"
    assert agent.requests == []
    assert env.state.list_trials("study") == []


def test_propose_trial_cancels_the_trial_when_the_agent_fails(env):
    """Report failure and cancel the provisional trial on an agent error."""
    context = env.make(agent=_FakeAgent(_failure()))

    result = ProposeTrialStage().run(context)

    assert result.status == "failure"
    assert "Proposal agent failed: boom" in result.message
    assert env.state.get_trial("trial_000001").status == "cancelled"


def test_propose_trial_drops_keys_outside_the_search_space(env):
    """Keep allowed keys and record removed ones in trial metadata."""
    atlas = {"search_space": {"allowed_keys": ["optimizer.*"]}}
    result, _, _ = _propose(
        env, {"optimizer.lr": 0.1, "trainer.max_epochs": 99}, atlas=atlas
    )

    assert result.status == "success"
    trial_dir = env.study_dir / "trials" / "trial_000001"
    meta = _load_yaml(trial_dir / "trial.yaml")
    assert meta["removed_keys"] == ["trainer.max_epochs"]
    training = _load_yaml(trial_dir / "resolved_training_config.yaml")
    assert training["trainer"]["max_epochs"] == 2


def test_propose_trial_fails_when_the_patch_is_emptied_by_validation(env):
    """Fail the proposal if every patched key is denied."""
    atlas = {"search_space": {"denied_keys": ["optimizer.*"]}}
    result, _, _ = _propose(env, {"optimizer.lr": 0.1}, atlas=atlas)

    assert result.status == "failure"
    assert "Patch became empty" in result.message
    assert env.state.get_trial("trial_000001").status == "cancelled"


def test_propose_trial_rejects_a_duplicate_hyperparameter_patch(env):
    """Cancel a proposal whose patch equals an earlier trial's patch."""
    first, _, _ = _propose(env)
    assert first.status == "success"

    second, _, _ = _propose(env)

    assert second.status == "duplicate"
    assert "same as trial_000001" in second.message
    assert env.state.get_trial("trial_000002").status == "cancelled"


def test_propose_trial_rejects_a_different_patch_with_the_same_config(env):
    """Detect duplicates by resolved config even when the patches differ."""
    _propose(env, {"optimizer.lr": 0.001})

    result, _, _ = _propose(env, {"optimizer.lr": 0.001, "optimizer.eps": 1.0e-8})

    assert result.status == "duplicate"
    assert result.payload["duplicate_of"] == "trial_000001"
    assert "Duplicate effective training config" in result.message


def test_propose_trial_ignores_unknown_extra_stages_and_routes_known_ones(env):
    """Filter extra stages and route to ``implement_code`` when requested."""
    agent = _FakeAgent(
        _proposal({"optimizer.lr": 0.5}, extra_stages=["bogus", "implement_code"])
    )
    context = env.make(agent=agent, atlas={"application": "architecture_search"})

    result = ProposeTrialStage().run(context)

    assert result.status == "implement_code"
    assert result.next_node == "implement_code"
    assert env.state.get_trial("trial_000001").extra_stages == ["implement_code"]
    request = agent.requests[0][0]
    assert "implement_code" in request.allowed_actions
    assert (
        "implement_code"
        in request.output_schema["properties"]["extra_stages"]["description"]
    )


def test_propose_trial_prefers_write_files_over_implement_code(env):
    """Route to ``write_files`` first when both extra stages are requested."""
    agent = _FakeAgent(
        _proposal({"optimizer.lr": 0.5}, extra_stages=["implement_code", "write_files"])
    )
    context = env.make(agent=agent)

    result = ProposeTrialStage().run(context)

    assert result.status == "write_files"
    assert result.next_node == "write_files"


def test_propose_trial_reuses_a_provisional_trial(env):
    """Continue the same trial id when a proposing trial has no config yet."""
    agent = _FakeAgent(_proposal({"optimizer.lr": 0.2}))
    context = env.make(agent=agent)
    stage = ProposeTrialStage()
    trial = stage._ensure_provisional_trial(context, None, [])
    assert trial.status == "proposing"
    context.current_trial = trial

    result = stage.run(context)

    assert result.status == "success"
    assert result.payload["trial_id"] == "trial_000001"
    assert len(env.state.list_trials("study")) == 1


def test_propose_trial_links_parent_and_inflight_patches(env):
    """Pass in-flight patches to the agent and parent the next trial."""
    _propose(env, {"optimizer.lr": 0.001})
    (env.recipe / "src-checklist.md").write_text("- check things")
    (env.study_dir / "knowledge" / "knowledge_pack.md").write_text("KNOW")

    _, _, agent = _propose(env, {"optimizer.lr": 0.002})

    request = agent.requests[0][0]
    assert request.inflight_patches == [{"optimizer.lr": 0.001}]
    assert "KNOW" in request.knowledge
    assert "- check things" in request.knowledge
    assert "trial_000001" in request.trial_history_csv
    assert env.state.get_trial("trial_000002").parent_trial_id == "trial_000001"


def test_propose_trial_marks_unreadable_base_configs_unavailable(env):
    """Show ``(unavailable)`` to the agent when a base config cannot load."""
    (env.recipe / "conf" / "metrics.yaml").unlink()
    agent = _FakeAgent(_proposal({"optimizer.lr": 0.3}))
    context = env.make(agent=agent)

    with pytest.raises(FileNotFoundError):
        ProposeTrialStage().run(context)

    assert "(unavailable)" in agent.requests[0][0].repo_context


def test_compute_config_signature_ignores_trial_paths_and_logger_names(tmp_path):
    """Normalize generated paths and logger names before comparing configs."""
    a = {"exp_dir": "x", "p": f"{tmp_path}/a", "trainer": {"logger": {"n": 1}}}
    b = {"exp_dir": "y", "p": f"{tmp_path}/a", "trainer": {"logger": {"n": 2}}}
    assert propose_trial._compute_config_signature(
        a, tmp_path
    ) == propose_trial._compute_config_signature(b, tmp_path)
    assert propose_trial._normalize_trial_config([f"{tmp_path}/z", 3], tmp_path) == [
        "<trial_dir>/z",
        3,
    ]


def test_propose_trial_helpers_compute_sequential_ids(tmp_path):
    """Number trials by how many already exist."""
    assert ProposeTrialStage()._compute_next_trial_id([1, 2]) == "trial_000003"
    assert Path(tmp_path).exists()


def test_propose_trial_fails_and_cancels_the_trial_when_the_agent_fails(tmp_path):
    from espnet3.bin.atlas.core.state import StateStore
    from espnet3.bin.atlas.stages.propose_trial import ProposeTrialStage

    class DownAgent:
        def run(self, request, artifact_dir=None):
            return AgentResponse(status="failure", message="agent down", content="")

    study_dir = tmp_path / "study"
    study_dir.mkdir()
    (study_dir / "program.md").write_text("Maximize valid_acc.", encoding="utf-8")
    state = StateStore(study_dir)
    state.create_or_load_study("study", study_dir)
    context = SimpleNamespace(
        study_dir=study_dir,
        recipe_dir=tmp_path / "recipe",
        config=OmegaConf.create(
            {
                "atlas": {
                    "application": "hyperparameter_search",
                    "metric": {"name": "valid_acc", "mode": "max"},
                    "recipe": {},
                    "search_space": {"allowed_keys": ["optimizer.*"]},
                }
            }
        ),
        scheduler=SimpleNamespace(should_stop=lambda: False, study_id="study"),
        state=state,
        agent=DownAgent(),
        current_trial=None,
    )

    result = ProposeTrialStage().run(context)

    assert result.status == "failure"
    assert "agent down" in result.message
    assert state.get_trial("trial_000001").status == "cancelled"

"""Tests for Atlas graph skills and coding-agent configuration."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from espnet3.bin.atlas.agents.cli import build_agent_config
from espnet3.bin.atlas.command import _default_graph_path
from espnet3.bin.atlas.core import AtlasError
from espnet3.bin.atlas.core.graph import load_graph
from espnet3.bin.atlas.core.models import (
    AgentRequest,
    AgentResponse,
    NodeConfig,
    ResearchGraph,
    Trial,
)
from espnet3.bin.atlas.core.scheduler import _find_skill_path, _SkillAgent
from espnet3.bin.atlas.core.stage import SkillStage


def _make_trial():
    """Build a minimal mutable trial fixture."""
    now = datetime.now(timezone.utc).isoformat()
    return Trial(
        trial_id="trial_000001",
        study_id="study",
        status="proposed",
        config_patch={},
        resolved_config_path=None,
        rationale="",
        expected_effect="",
        risk="",
        score=None,
        score_name=None,
        decision=None,
        parent_trial_id=None,
        attempt_count=0,
        created_at=now,
        updated_at=now,
    )


def test_agent_config_selects_provider_and_preserves_cli_options():
    """Parse provider-specific CLI settings without losing local-model options."""
    config = build_agent_config(
        {
            "provider": "codex",
            "codex": {
                "model": "shogun-worker",
                "config": {"model_provider": "vllm", "web_search": "live"},
                "args": ["--full-auto"],
            },
            "env": {"OPENAI_API_KEY": "local-key"},
        }
    )

    assert config.provider == "codex"
    assert config.codex.model == "shogun-worker"
    assert config.codex.config["model_provider"] == "vllm"
    assert config.codex.config["web_search"] == "live"
    assert config.codex.args == ["--full-auto"]
    assert config.env == {"OPENAI_API_KEY": "local-key"}


@pytest.mark.parametrize("provider", ["", "openai"])
def test_agent_config_rejects_unsupported_provider(provider):
    """Reject missing or unsupported agent selections with a clear error."""
    with pytest.raises(ValueError, match="provider must be one of 'codex', 'claude', 'command'"):
        build_agent_config({"provider": provider})


def test_agent_config_accepts_claude_options():
    """Parse Claude-specific options and normalize list values."""
    config = build_agent_config(
        OmegaConf.create(
            {
                "provider": "claude",
                "claude": {
                    "model": "sonnet",
                    "allowed_tools": ["Read", "Bash"],
                    "max_turns": 5,
                },
            }
        )
    )
    assert config.provider == "claude"
    assert config.claude.model == "sonnet"
    assert config.claude.allowed_tools == ["Read", "Bash"]
    assert config.claude.max_turns == 5
    assert build_agent_config({"provider": "Claude"}).provider == "claude"


def test_load_graph_loads_an_explicit_skill_stage(tmp_path):
    """Keep ``skill`` as node config while ``target`` names ``SkillStage``."""
    graph_path = tmp_path / "graph.yaml"
    graph_path.write_text(
        "name: skills\nstart: analyze\nnodes:\n  analyze:\n"
        "    target: espnet3.bin.atlas.core.stage.SkillStage\n"
        "    skill: hyperparameter_search_analyze\n    terminal: true\n",
        encoding="utf-8",
    )

    graph = load_graph(graph_path)

    assert graph.nodes["analyze"].target.endswith("core.stage.SkillStage")
    assert graph.nodes["analyze"].skill == "hyperparameter_search_analyze"


def test_load_graph_requires_a_target_even_for_a_skill_node(tmp_path):
    """A ``skill`` alone no longer selects ``SkillStage``."""
    graph_path = tmp_path / "graph.yaml"
    graph_path.write_text(
        "name: skills\nstart: analyze\nnodes:\n  analyze:\n"
        "    skill: hyperparameter_search_analyze\n    terminal: true\n",
        encoding="utf-8",
    )

    with pytest.raises(AtlasError, match="Node 'analyze' needs a `target`"):
        load_graph(graph_path)


def test_load_graph_requires_a_target_for_a_node_without_one(tmp_path):
    """Reject a node that does not identify an implementation."""
    graph_path = tmp_path / "graph.yaml"
    graph_path.write_text("start: empty\nnodes:\n  empty:\n    terminal: true\n")

    with pytest.raises(AtlasError, match="Node 'empty' needs a `target`"):
        load_graph(graph_path)


def test_packaged_graphs_name_a_target_on_every_node():
    """Every shipped graph loads, so none relies on an implicit ``SkillStage``."""
    for application in ("hyperparameter_search", "architecture_search"):
        graph = load_graph(_default_graph_path(application))
        assert graph.nodes
        assert all(node.target for node in graph.nodes.values())


def test_find_skill_path_rejects_path_traversal_and_finds_parent_skill(tmp_path):
    """Resolve skills from recipe ancestors while rejecting unsafe names."""
    recipe_dir = tmp_path / "egs3" / "recipe"
    recipe_dir.mkdir(parents=True)
    skill_path = tmp_path / ".agents" / "skills" / "example" / "SKILL.md"
    skill_path.parent.mkdir(parents=True)
    skill_path.write_text("Instructions", encoding="utf-8")

    assert _find_skill_path(recipe_dir, "example") == skill_path
    for unsafe_name in ("../secret", "nested/skill", "nested\\skill"):
        with pytest.raises(ValueError, match="Invalid skill name"):
            _find_skill_path(recipe_dir, unsafe_name)
    with pytest.raises(FileNotFoundError, match="not found"):
        _find_skill_path(recipe_dir, "missing")


def test_skill_agent_injects_instructions_without_mutating_request(tmp_path):
    """Add the selected SKILL.md to a copied agent request."""
    skill_path = tmp_path / "SKILL.md"
    skill_path.write_text("Follow these instructions", encoding="utf-8")
    request = AgentRequest(
        task="task",
        objective="objective",
        knowledge="",
        repo_context="",
        trial_history_csv="",
        latest_metrics={},
        context_files={"existing": "value"},
        allowed_actions=[],
        output_schema={},
    )

    class RecordingAgent:
        """Capture the forwarded request for assertions."""

        def run(self, forwarded, artifact_dir=None):
            self.request = forwarded
            self.artifact_dir = artifact_dir
            return "response"

    agent = RecordingAgent()
    result = _SkillAgent(agent, skill_path).run(request, artifact_dir=tmp_path)

    assert result == "response"
    assert agent.request.context_files == {
        "existing": "value",
        "skill_instructions": "Follow these instructions",
    }
    assert request.context_files == {"existing": "value"}
    assert agent.artifact_dir == tmp_path


def _stage_context(tmp_path, *, node, agent, trial=None, state=None):
    """Construct the lightweight context consumed by SkillStage."""
    recipe_dir = tmp_path / "recipe"
    recipe_dir.mkdir(exist_ok=True)
    study_dir = tmp_path / "study"
    study_dir.mkdir(exist_ok=True)
    graph = ResearchGraph("test", 1, node.name, {node.name: node})
    return SimpleNamespace(
        graph=graph,
        node_name=node.name,
        current_trial=trial,
        config=OmegaConf.create({"atlas": {"code_edit": {}}}),
        study_dir=study_dir,
        recipe_dir=recipe_dir,
        scheduler=SimpleNamespace(should_stop=lambda: False),
        agent=agent,
        state=state or SimpleNamespace(update_trial=lambda _: None),
        trial_dir=lambda: study_dir / "trials" / trial.trial_id,
        artifact_dir=lambda: study_dir / "node_runs" / "run",
    )


def test_skill_stage_writes_output_and_only_updates_allowed_trial_fields(tmp_path):
    """Persist the requested output and protect trial identity fields."""

    class SuccessAgent:
        """Return a deterministic successful skill result."""

        def run(self, request, artifact_dir=None):
            self.request = request
            return AgentResponse(
                status="success",
                message="analysis complete",
                content="",
                thread_id="thread-1",
                structured={
                    "status": "success",
                    "report": "# Result\n",
                    "trial_updates": {"score": 0.2, "trial_id": "tampered"},
                },
            )

    node = NodeConfig(
        name="analyze",
        target="espnet3.bin.atlas.core.stage.SkillStage",
        execution="local",
        resources=None,
        skill="hyperparameter_search_analyze",
        output_file="reports/{trial_id}.md",
        output_key="report",
    )
    trial = _make_trial()
    updates = []
    context = _stage_context(
        tmp_path,
        node=node,
        agent=SuccessAgent(),
        trial=trial,
        state=SimpleNamespace(update_trial=updates.append),
    )

    result = SkillStage().run(context)

    report_path = tmp_path / "study" / "reports" / "trial_000001.md"
    assert result.status == "success"
    assert report_path.read_text(encoding="utf-8") == "# Result\n"
    assert result.artifacts["report"] == str(report_path)
    assert trial.trial_id == "trial_000001"
    assert trial.score == 0.2
    assert trial.codex_thread_id == "thread-1"
    assert updates == [trial]


def test_skill_stage_rejects_output_path_escape_before_agent_call(tmp_path):
    """Reject output paths escaping the study directory before execution."""

    class UnexpectedAgent:
        """Fail if unsafe output configuration reaches an agent."""

        def run(self, request, artifact_dir=None):
            pytest.fail("agent must not run for an unsafe output path")

    node = NodeConfig(
        name="analyze",
        target="espnet3.bin.atlas.core.stage.SkillStage",
        execution="local",
        resources=None,
        skill="analysis",
        output_file="../outside.md",
    )
    result = SkillStage().run(
        _stage_context(tmp_path, node=node, agent=UnexpectedAgent())
    )
    assert result.status == "failure"
    assert "stay in study_dir" in result.message


def test_skill_stage_reports_missing_required_output_and_supports_fallback(tmp_path):
    """Fail missing required output or write its configured fallback."""

    class EmptyAgent:
        """Return success without the required output field."""

        def run(self, request, artifact_dir=None):
            return AgentResponse(
                status="success", message="", content="", structured={"summary": "done"}
            )

    base = dict(
        name="analyze",
        target="espnet3.bin.atlas.core.stage.SkillStage",
        execution="local",
        resources=None,
        skill="analysis",
        output_file="reports/result.md",
        output_key="report",
    )
    node = NodeConfig(**base)
    missing = SkillStage().run(_stage_context(tmp_path, node=node, agent=EmptyAgent()))
    assert missing.status == "failure"
    assert "did not return required field 'report'" in missing.message

    node.fallback_output = "Fallback report\n"
    fallback = SkillStage().run(_stage_context(tmp_path, node=node, agent=EmptyAgent()))
    assert fallback.status == "terminal"
    assert (
        tmp_path / "study" / "reports" / "result.md"
    ).read_text() == "Fallback report\n"

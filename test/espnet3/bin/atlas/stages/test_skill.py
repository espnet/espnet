"""Tests for skill."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from espnet3.bin.atlas.core.graph import load_graph
from espnet3.bin.atlas.core.models import (
    AgentResponse,
    NodeConfig,
    ResearchGraph,
    Trial,
)
from espnet3.bin.atlas.stages.skill import SkillStage


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
        get_trial_dir=lambda: study_dir / "trials" / trial.trial_id,
        get_artifact_dir=lambda: study_dir / "node_runs" / "run",
    )


def _graph_path(application):
    """Return the path of a packaged graph YAML."""
    from importlib import resources

    resource = resources.files("espnet3.bin.atlas").joinpath(
        "graphs", f"{application}.yaml"
    )
    with resources.as_file(resource) as path:
        return Path(path)


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
                    "message": "done",
                    "report": "# Result\n",
                    "score": 0.2,
                    "trial_id": "tampered",
                },
            )

    node = NodeConfig(
        name="analyze",
        target="espnet3.bin.atlas.stages.skill.SkillStage",
        execution="local",
        resources=None,
        skill="atlas_hyperparameter_search_analyze",
        output_file="reports/{trial_id}.md",
        output_key="report",
        trial_update_keys=["score", "trial_id"],
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
        target="espnet3.bin.atlas.stages.skill.SkillStage",
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


def test_skill_stage_writes_its_fallback_when_the_agent_fails(tmp_path):
    """Fail the node, or write the configured fallback, after an agent failure."""

    class FailingAgent:
        def run(self, request, artifact_dir=None):
            return AgentResponse(status="failure", message="boom", content="")

    base = dict(
        name="analyze",
        target="espnet3.bin.atlas.stages.skill.SkillStage",
        execution="local",
        resources=None,
        skill="analysis",
        output_file="reports/result.md",
        output_key="report",
    )
    node = NodeConfig(**base)
    failed = SkillStage().run(_stage_context(tmp_path, node=node, agent=FailingAgent()))
    assert failed.status == "failure"

    node.fallback_output = "Fallback report\n"
    fallback = SkillStage().run(
        _stage_context(tmp_path, node=node, agent=FailingAgent())
    )
    assert fallback.status == "terminal"
    assert (
        tmp_path / "study" / "reports" / "result.md"
    ).read_text() == "Fallback report\n"


def test_reply_schema_is_closed_and_takes_its_status_from_the_node_routes():
    from espnet3.bin.atlas.stages.skill import build_reply_schema

    node = NodeConfig(
        name="reflect",
        target="espnet3.bin.atlas.stages.skill.SkillStage",
        execution="local",
        resources=None,
        output_schema={"reflection": {"type": "string"}},
        on={"continue": "a", "done": "b", "failure": "a"},
    )

    schema = build_reply_schema(node)

    assert schema["additionalProperties"] is False
    assert schema["required"] == ["status", "message", "reflection"]
    assert schema["properties"]["status"]["enum"] == ["continue", "done", "failure"]
    node.on = {}
    assert build_reply_schema(node)["properties"]["status"]["enum"] == ["terminal"]


def test_skill_stage_appends_instead_of_replacing_when_asked(tmp_path):
    """Keep the existing pack and add the agent's entry after a blank line."""

    class EntryAgent:
        def run(self, request, artifact_dir=None):
            return AgentResponse(
                status="success",
                message="",
                content="",
                structured={
                    "status": "success",
                    "message": "",
                    "entry": "## Trial 2\n- ok\n",
                },
            )

    node = NodeConfig(
        name="update_knowledge",
        target="espnet3.bin.atlas.stages.skill.SkillStage",
        execution="local",
        resources=None,
        skill="atlas_hyperparameter_search_update_knowledge",
        output_file="knowledge/knowledge_pack.md",
        output_key="entry",
        output_append=True,
    )
    context = _stage_context(tmp_path, node=node, agent=EntryAgent())
    pack = tmp_path / "study" / "knowledge" / "knowledge_pack.md"
    pack.parent.mkdir(parents=True)
    pack.write_text("# Pack\n\n## Trial 1\n- first\n", encoding="utf-8")

    result = SkillStage().run(context)

    assert result.status == "success"
    assert pack.read_text(encoding="utf-8") == (
        "# Pack\n\n## Trial 1\n- first\n\n## Trial 2\n- ok\n"
    )


def test_research_context_failure_still_leaves_a_knowledge_pack(tmp_path):
    class DownAgent:
        def run(self, request, artifact_dir=None):
            return AgentResponse(status="failure", message="agent down", content="")

    node = load_graph(_graph_path("hyperparameter_search")).nodes["research_context"]
    context = _stage_context(tmp_path, node=node, agent=DownAgent())

    result = SkillStage().run(context)

    pack = tmp_path / "study" / "knowledge" / "knowledge_pack.md"
    assert result.status == "terminal"
    assert pack.read_text(encoding="utf-8").startswith("# Knowledge Pack")

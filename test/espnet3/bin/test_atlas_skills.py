"""Tests for Atlas graph skills and coding-agent configuration."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from espnet3.bin.atlas.agents.cli import build_agent_config
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
from espnet3.bin.atlas.stages.skill import SkillStage


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


def _graph_path(application):
    """Return the path of a packaged graph YAML."""
    from importlib import resources

    resource = resources.files("espnet3.bin.atlas").joinpath(
        "graphs", f"{application}.yaml"
    )
    with resources.as_file(resource) as path:
        return Path(path)


def test_load_graph_loads_an_explicit_skill_stage(tmp_path):
    """Keep ``skill`` as node config while ``target`` names ``SkillStage``."""
    graph_path = tmp_path / "graph.yaml"
    graph_path.write_text(
        "name: skills\nstart: analyze\nnodes:\n  analyze:\n"
        "    target: espnet3.bin.atlas.stages.skill.SkillStage\n"
        "    skill: hyperparameter_search_analyze\n    terminal: true\n",
        encoding="utf-8",
    )

    graph = load_graph(graph_path)

    assert graph.nodes["analyze"].target.endswith("stages.skill.SkillStage")
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
        graph = load_graph(_graph_path(application))
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
        get_trial_dir=lambda: study_dir / "trials" / trial.trial_id,
        get_artifact_dir=lambda: study_dir / "node_runs" / "run",
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
        target="espnet3.bin.atlas.stages.skill.SkillStage",
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
        target="espnet3.bin.atlas.stages.skill.SkillStage",
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


def test_skill_stage_appends_instead_of_replacing_when_asked(tmp_path):
    """Keep the existing pack and add the agent's entry after a blank line."""

    class EntryAgent:
        def run(self, request, artifact_dir=None):
            return AgentResponse(
                status="success",
                message="",
                content="",
                structured={"status": "success", "entry": "## Trial 2\n- ok\n"},
            )

    node = NodeConfig(
        name="update_knowledge",
        target="espnet3.bin.atlas.stages.skill.SkillStage",
        execution="local",
        resources=None,
        skill="hyperparameter_search_update_knowledge",
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


def test_check_budget_routes_done_only_when_the_budget_is_spent():
    from espnet3.bin.atlas.stages.check_budget import CheckBudgetStage

    for spent, expected in ((False, "continue"), (True, "done")):
        context = SimpleNamespace(scheduler=SimpleNamespace(should_stop=lambda: spent))
        assert CheckBudgetStage().run(context).status == expected


def test_every_skill_named_by_a_packaged_graph_exists():
    """A graph must not name a skill that has no SKILL.md to run."""
    repo_root = Path(__file__).resolve().parents[3]
    for application in ("hyperparameter_search", "architecture_search"):
        graph = load_graph(_graph_path(application))
        for node in graph.nodes.values():
            if node.skill:
                assert _find_skill_path(repo_root, node.skill).is_file(), node.skill


def test_update_knowledge_is_a_skill_followed_by_the_budget_check():
    for application in ("hyperparameter_search", "architecture_search"):
        nodes = load_graph(_graph_path(application)).nodes
        assert nodes["update_knowledge"].skill == (
            "hyperparameter_search_update_knowledge"
        )
        assert nodes["update_knowledge"].output_append is True
        assert nodes["update_knowledge"].on == {
            "success": "check_budget",
            "failure": "check_budget",
        }
        assert nodes["check_budget"].on == {
            "continue": "propose_trial",
            "done": "select_best",
        }


def test_research_context_is_a_skill_that_writes_the_first_pack_with_a_fallback():
    for application in ("hyperparameter_search", "architecture_search"):
        graph = load_graph(_graph_path(application))
        node = graph.nodes["research_context"]

        assert node.skill == "hyperparameter_search_research_context"
        assert node.output_file == "knowledge/knowledge_pack.md"
        assert node.output_key == "pack" and not node.output_append
        assert node.fallback_output.startswith("# Knowledge Pack")
        # A fallback write returns status "terminal"; it must still route on.
        assert node.on == {
            "success": "propose_trial",
            "failure": "propose_trial",
            "terminal": "propose_trial",
        }


def test_research_context_failure_still_leaves_a_knowledge_pack(tmp_path):
    class DownAgent:
        def run(self, request, artifact_dir=None):
            return AgentResponse(status="failure", message="agent down", content="")

    node = load_graph(_graph_path("hyperparameter_search")).nodes[
        "research_context"
    ]
    context = _stage_context(tmp_path, node=node, agent=DownAgent())

    result = SkillStage().run(context)

    pack = tmp_path / "study" / "knowledge" / "knowledge_pack.md"
    assert result.status == "terminal"
    assert pack.read_text(encoding="utf-8").startswith("# Knowledge Pack")


def test_packaged_graphs_route_only_to_defined_nodes_and_end_in_analyze():
    for application in ("hyperparameter_search", "architecture_search"):
        graph = load_graph(_graph_path(application))

        assert "summarize" not in graph.nodes
        for node in graph.nodes.values():
            for target in node.on.values():
                assert target in graph.nodes, (application, node.name, target)
        assert graph.nodes["select_best"].on == {
            "success": "analyze",
            "failure": "analyze",
        }
        assert graph.nodes["analyze"].terminal is True


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


def _init_args(tmp_path, application_line):
    import argparse

    recipe = tmp_path / "egs3" / "corpus" / "esp2_asr"
    (recipe / "conf").mkdir(parents=True)
    (recipe / "conf" / "atlas.yaml").write_text(
        "atlas:\n  study_name: s\n  study_dir: exp/atlas/s\n"
        + application_line
        + "  metric: {name: m, mode: min}\n",
        encoding="utf-8",
    )
    return recipe, argparse.Namespace(
        recipe_dir=str(recipe), config=str(recipe / "conf" / "atlas.yaml")
    )


def test_init_requires_a_known_application(tmp_path):
    from espnet3.bin.atlas.command import handle_init

    cases = (("a", "", "''"), ("b", "  application: nope\n", "'nope'"))
    for name, line, bad in cases:
        _, args = _init_args(tmp_path / name, line)
        with pytest.raises(ValueError) as error:
            handle_init(args)
        assert "architecture_search, hyperparameter_search" in str(error.value)
        assert bad in str(error.value)


def test_init_copies_the_packaged_graph_named_by_application(tmp_path):
    from espnet3.bin.atlas.command import handle_init

    recipe, args = _init_args(tmp_path, "  application: hyperparameter_search\n")

    assert handle_init(args) == 0
    study = recipe / "exp" / "atlas" / "s"
    assert (study / "graph.yaml").read_bytes() == (
        _graph_path("hyperparameter_search").read_bytes()
    )
    assert (study / "program.md").is_file() and (study / "atlas.yaml").is_file()
    assert all((study / name).is_dir() for name in ("knowledge", "trials", "node_runs"))

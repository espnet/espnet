"""End-to-end Atlas CLI tests with a deterministic dummy agent."""

from __future__ import annotations

import csv
import json

from espnet3.bin.atlas import command
from espnet3.bin.atlas.core import InvalidAgentResponseError
from espnet3.bin.atlas.core.models import AgentResponse
from espnet3.bin.atlas.core.state import StateStore
from espnet3.bin.main import main as run_cli


class _DummyCodingAgent:
    """Return a stable result while recording requests from the CLI runtime."""

    def __init__(self):
        self.requests = []

    def run(self, request, artifact_dir=None):
        self.requests.append((request, artifact_dir))
        return AgentResponse(
            status="success",
            message="dummy analysis complete",
            content="",
            thread_id="dummy-session",
            structured={
                "status": "terminal",
                "message": "study finished",
                "summary": "integration passed",
                "report": "# Dummy report\n",
            },
        )


def _write_skill_study(tmp_path):
    """Write a one-node skill graph recipe and return its directory and config."""
    # Config discovery infers the TEMPLATE defaults from the egs3/<corpus>/<system>
    # layout.
    recipe_dir = tmp_path / "egs3" / "integration" / "esp2_asr"
    recipe_dir.mkdir(parents=True)
    skill_file = recipe_dir / ".agents/skills/dummy_analysis/SKILL.md"
    skill_file.parent.mkdir(parents=True)
    skill_file.write_text("Use the provided study context.", encoding="utf-8")

    config_path = recipe_dir / "conf" / "atlas.yaml"
    config_path.parent.mkdir()
    config_path.write_text(
        "atlas:\n"
        "  study_name: integration-study\n"
        "  study_dir: exp/atlas/integration-study\n"
        "  application: graph.yaml\n"
        "  objective_file: program.md\n"
        "  metric:\n"
        "    name: score\n"
        "    mode: min\n"
        "  budget:\n"
        "    max_trials: 1\n"
        "    max_failures: 1\n"
        "  tick_interval_sec_idle: 1\n"
        "  submission:\n"
        "    resources: {}\n"
        "  agent:\n"
        "    provider: codex\n",
        encoding="utf-8",
    )
    graph_path = recipe_dir / "graph.yaml"
    graph_path.write_text(
        "name: dummy-integration\n"
        "version: 1\n"
        "start: analyze\n"
        "nodes:\n"
        "  analyze:\n"
        "    target: espnet3.bin.atlas.stages.skill.SkillStage\n"
        "    execution: local\n"
        "    skill: dummy_analysis\n"
        "    output_file: reports/analysis.md\n"
        "    output_key: report\n"
        "    output_schema:\n"
        "      report: {type: string}\n"
        "      summary: {type: string}\n"
        "    terminal: true\n",
        encoding="utf-8",
    )
    return recipe_dir, config_path


def test_cli_runs_skill_graph_to_completion_with_dummy_agent(tmp_path, monkeypatch):
    """Run init and run commands through a skill graph without external agents."""
    recipe_dir, config_path = _write_skill_study(tmp_path)

    dummy_agent = _DummyCodingAgent()
    monkeypatch.setattr(command, "_build_agent", lambda _config, _recipe: dummy_agent)
    monkeypatch.setattr(
        "sys.argv",
        [
            "espnet3",
            "atlas",
            "init",
            "--recipe-dir",
            str(recipe_dir),
            "--config",
            str(config_path),
        ],
    )
    run_cli()

    study_dir = recipe_dir / "exp/atlas/integration-study"
    monkeypatch.setattr(
        "sys.argv",
        [
            "espnet3",
            "atlas",
            "run",
            "--study-dir",
            str(study_dir),
            "--recipe-dir",
            str(recipe_dir),
            "--poll-interval",
            "0.01",
        ],
    )
    run_cli()

    assert len(dummy_agent.requests) == 1
    request, artifact_dir = dummy_agent.requests[0]
    assert request.task == "dummy_analysis"
    assert (
        request.context_files["skill_instructions"] == "Use the provided study context."
    )
    assert artifact_dir == study_dir / "node_runs" / next(
        row["run_id"]
        for row in StateStore(study_dir).list_node_runs("integration-study")
    )
    assert (study_dir / "reports/analysis.md").read_text(
        encoding="utf-8"
    ) == "# Dummy report\n"

    state = StateStore(study_dir)
    assert state.get_study("integration-study").status == "completed"
    node_runs = state.list_node_runs("integration-study")
    assert len(node_runs) == 1
    assert node_runs[0]["status"] == "completed"
    assert node_runs[0]["result_status"] == "terminal"
    assert state.get_agent_session_id("integration-study") == "dummy-session"
    assert state.is_agent_context_initialized("integration-study") is True
    with (study_dir / "state_events.csv").open(newline="", encoding="utf-8") as handle:
        events = list(csv.DictReader(handle))
    assert [json.loads(event["event_type"]) for event in events] == [
        "node_started",
        "node_finished",
        "study_completed",
    ]
    assert json.loads(json.loads(events[-1]["payload_json"]))["status"] == "terminal"


class _InvalidReplyAgent:
    """Fail the way the session does when a reply violates its schema."""

    def run(self, request, artifact_dir=None):
        raise InvalidAgentResponseError("agent reply violated its schema")


def test_invalid_agent_response_is_recorded_as_a_failed_node_run(tmp_path, monkeypatch):
    recipe_dir, config_path = _write_skill_study(tmp_path)
    monkeypatch.setattr(
        command, "_build_agent", lambda _config, _recipe: _InvalidReplyAgent()
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "espnet3",
            "atlas",
            "init",
            "--recipe-dir",
            str(recipe_dir),
            "--config",
            str(config_path),
        ],
    )
    run_cli()
    study_dir = recipe_dir / "exp/atlas/integration-study"

    # One tick, not `atlas run`: a failed node run schedules nothing further, so
    # the run loop would wait for work that never comes.
    command._load_runtime(study_dir, recipe_dir).tick()

    study_dir = recipe_dir / "exp/atlas/integration-study"
    state = StateStore(study_dir)
    (node_run,) = state.list_node_runs("integration-study")
    assert node_run["result_status"] == "failure"
    result = json.loads(node_run["result_payload_json"])
    assert result["payload"]["invalid_agent_response"] is True
    assert "violated its schema" in result["message"]
    failure_file = study_dir / "node_runs" / node_run["run_id"] / ".stage_failure.json"
    assert json.loads(failure_file.read_text(encoding="utf-8"))["status"] == "failure"

"""Tests for the coding-agent providers and how CliAgent runs them."""

from __future__ import annotations

import json
import stat
import sys

import pytest

from espnet3.bin.atlas.agents.cli import CliAgent, build_agent_config, render_prompt
from espnet3.bin.atlas.agents.providers import create_provider
from espnet3.bin.atlas.core.models import AgentRequest


def _request(**changes):
    values = dict(
        task="t",
        objective="do it",
        knowledge="",
        repo_context="",
        trial_history_csv="",
        latest_metrics={},
        context_files={},
        allowed_actions=["read_only"],
        output_schema={},
    )
    values.update(changes)
    return AgentRequest(**values)


def _provider(config):
    return create_provider(build_agent_config(config))


def _lines(*events):
    return "\n".join(json.dumps(event) for event in events)


def test_claude_provider_reads_the_result_event_and_session(tmp_path):
    parsed = _provider({"provider": "claude"}).parse(
        _lines(
            {"type": "system", "session_id": "s1"},
            {"type": "result", "is_error": False, "result": '{"a": 1}'},
        ),
        tmp_path / "out.txt",
    )

    assert (parsed.content, parsed.thread_id, parsed.error) == ('{"a": 1}', "s1", None)


def test_claude_provider_prefers_native_structured_output(tmp_path):
    parsed = _provider({"provider": "claude"}).parse(
        _lines({"type": "result", "structured_output": {"x": 1}, "result": "prose"}),
        tmp_path / "out.txt",
    )

    assert json.loads(parsed.content) == {"x": 1}


@pytest.mark.parametrize(
    "events, message",
    [
        ([], "no final result event"),
        (
            [{"type": "result", "is_error": True, "subtype": "error_max_turns"}],
            "max_turns",
        ),
    ],
)
def test_claude_provider_reports_a_failed_stream(tmp_path, events, message):
    parsed = _provider({"provider": "claude"}).parse(_lines(*events), tmp_path / "o")

    assert message in parsed.error


def test_codex_provider_falls_back_to_the_last_agent_message(tmp_path):
    output = tmp_path / "out.txt"
    output.write_text("")
    provider = _provider({"provider": "codex"})
    stdout = _lines(
        {"type": "thread.started", "thread_id": "th1"},
        {"type": "item.completed", "item": {"type": "agent_message", "text": " hi "}},
    )

    parsed = provider.parse(stdout, output)

    assert (parsed.content, parsed.thread_id) == ("hi", "th1")
    output.write_text("from file\n")
    assert provider.parse(stdout, output).content == "from file"


def test_codex_provider_records_token_usage(tmp_path):
    provider = _provider({"provider": "codex"})

    provider.record_usage(
        tmp_path,
        _request(),
        ["codex", "exec"],
        _lines({"type": "turn.completed", "usage": {"input_tokens": 3}}),
    )

    summary = json.loads((tmp_path / "agent_usage.json").read_text())
    assert summary["last_usage"] == {"input_tokens": 3}
    assert (tmp_path / "agent_usage_history.jsonl").is_file()


def test_unknown_provider_is_rejected():
    with pytest.raises(ValueError, match="provider must be one of"):
        build_agent_config({"provider": "vibe"})


def test_render_prompt_shows_the_schema():
    prompt = render_prompt(_request(output_schema={"type": "object"}))

    assert prompt.startswith("Task: t\n\nFollow the requested output shape exactly.")
    assert '"type": "object"' in prompt


def _fake_claude(tmp_path, reply):
    """A stand-in ``claude`` that prints a result event carrying ``reply``."""
    script = tmp_path / "fake_claude.py"
    script.write_text(
        "#!" + sys.executable + "\n"
        "import json, sys\n"
        "prompt = sys.stdin.read()\n"
        "assert 'do it' in prompt\n"
        "assert '--json-schema' in sys.argv\n"
        "print(json.dumps({'type': 'result', 'session_id': 's1', 'structured_output': "
        + repr(reply)
        + "}))\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script


def test_cli_agent_returns_the_schema_constrained_reply(tmp_path):
    from espnet3.bin.atlas.resources import load_schema

    reply = {"should_stop": False, "reason": "ok", "key_metrics": []}
    agent = CliAgent(
        {"provider": "claude", "claude": {"executable": str(_fake_claude(tmp_path, reply))}},
        recipe_dir=tmp_path,
    )

    response = agent.run(
        _request(output_schema=load_schema("early_stop")),
        artifact_dir=tmp_path / "artifacts",
    )

    assert response.status == "success"
    assert response.structured == reply
    assert response.thread_id == "s1"
    assert (tmp_path / "artifacts" / "agent_schema.json").is_file()


def test_cli_agent_fails_a_reply_that_violates_the_schema(tmp_path):
    from espnet3.bin.atlas.resources import load_schema

    reply = {"should_stop": "maybe", "reason": "ok", "key_metrics": []}
    agent = CliAgent(
        {"provider": "claude", "claude": {"executable": str(_fake_claude(tmp_path, reply))}},
        recipe_dir=tmp_path,
    )

    response = agent.run(
        _request(output_schema=load_schema("early_stop")),
        artifact_dir=tmp_path / "artifacts",
    )

    assert response.status == "failure"
    assert "should_stop" in response.message
    assert "parse_error" in response.artifacts


def test_cli_agent_reports_a_missing_executable(tmp_path):
    agent = CliAgent(
        {"provider": "claude", "claude": {"executable": "no-such-agent-cli"}},
        recipe_dir=tmp_path,
    )

    response = agent.run(_request(), artifact_dir=tmp_path / "artifacts")

    assert response.status == "failure"
    assert "no-such-agent-cli" in response.message

"""Tests for the coding-agent providers and the generic ``command`` provider."""

from __future__ import annotations

import json
import stat
import sys
from pathlib import Path

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


def test_render_prompt_adds_the_json_contract_only_with_a_schema():
    plain = render_prompt(_request())
    with_schema = render_prompt(_request(output_schema={"type": "object"}))

    assert "FINAL RESPONSE CONTRACT" not in plain
    assert "FINAL RESPONSE CONTRACT" in with_schema
    assert plain.startswith("Task: t\n\nFollow the requested output shape exactly.")


def test_command_provider_sends_the_prompt_on_stdin_by_default():
    invocation = _provider(
        {"provider": "command", "command": {"argv": ["my-agent", "--quiet"]}}
    ).build_command(_request(), "PROMPT", cwd=Path("."), output_path=Path("o"))

    assert invocation.argv == ["my-agent", "--quiet"]
    assert invocation.stdin == "PROMPT"


def test_command_provider_can_put_the_prompt_in_an_argument():
    invocation = _provider(
        {
            "provider": "command",
            "command": {"argv": ["my-agent", "--prompt={prompt}"], "prompt_via": "arg"},
        }
    ).build_command(_request(), "PROMPT", cwd=Path("."), output_path=Path("o"))

    assert invocation.argv == ["my-agent", "--prompt=PROMPT"]
    assert invocation.stdin is None


@pytest.mark.parametrize(
    "command, message",
    [
        ({"argv": []}, "argv must name the program"),
        ({"argv": ["x"], "prompt_via": "file"}, "prompt_via must be"),
        ({"argv": ["x"], "prompt_via": "arg"}, "needs a {prompt} argument"),
    ],
)
def test_command_provider_rejects_incomplete_settings(command, message):
    provider = _provider({"provider": "command", "command": command})

    with pytest.raises(ValueError, match=message):
        provider.build_command(_request(), "P", cwd=Path("."), output_path=Path("o"))


def test_cli_agent_runs_any_command_that_prints_json(tmp_path):
    script = tmp_path / "fake_agent.py"
    script.write_text(
        "import sys\n"
        "prompt = sys.stdin.read()\n"
        "assert 'do it' in prompt\n"
        "print('{\"should_stop\": false, \"reason\": \"ok\", \"key_metrics\": {}}')\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    from espnet3.bin.atlas.resources import load_schema

    agent = CliAgent(
        {"provider": "command", "command": {"argv": [sys.executable, str(script)]}},
        recipe_dir=tmp_path,
    )

    response = agent.run(
        _request(output_schema=load_schema("early_stop")),
        artifact_dir=tmp_path / "artifacts",
    )

    assert response.status == "success"
    assert response.structured["should_stop"] is False
    assert response.thread_id is None


def test_cli_agent_reports_a_missing_executable(tmp_path):
    agent = CliAgent(
        {"provider": "command", "command": {"argv": ["no-such-agent-cli"]}},
        recipe_dir=tmp_path,
    )

    response = agent.run(_request(), artifact_dir=tmp_path / "artifacts")

    assert response.status == "failure"
    assert "no-such-agent-cli" in response.message

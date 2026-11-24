"""Tests for claude."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from espnet3.bin.atlas.agents.cli import build_agent_config
from espnet3.bin.atlas.agents.providers import create_provider
from espnet3.bin.atlas.agents.providers.claude import _extract_content
from espnet3.bin.atlas.core.models import AgentRequest
from espnet3.bin.atlas.resources import load_schema


def _request(**changes):
    values = dict(
        task="t",
        objective="",
        knowledge="",
        repo_context="",
        trial_history_csv="",
        latest_metrics={},
        context_files={},
        allowed_actions=[],
        output_schema={},
    )
    values.update(changes)
    return AgentRequest(**values)


def _build(request, **claude):
    provider = create_provider(
        build_agent_config({"provider": "claude", "claude": claude})
    )
    return provider.build_command(
        request,
        "prompt",
        cwd=Path("/w"),
        output_path=Path("/o"),
        schema_path=Path("/s"),
    )


def _argv(provider_config, request):
    provider = create_provider(build_agent_config(provider_config))
    return provider.build_command(
        request,
        "PROMPT",
        cwd=Path("."),
        output_path=Path("out.txt"),
        schema_path=Path("schema.json"),
    ).argv


def _request_old(schema):
    return AgentRequest(
        task="t",
        objective="o",
        knowledge="",
        repo_context="",
        trial_history_csv="",
        latest_metrics={},
        context_files={},
        allowed_actions=["read_only"],
        output_schema=schema,
    )


def _lines(*events):
    return "\n".join(json.dumps(event) for event in events)


def _provider(config):
    return create_provider(build_agent_config(config))


def test_build_command_defaults():
    """Print stream-json with the default permission mode and no extras."""
    inv = _build(_request())

    assert inv.stdin == "prompt"
    assert inv.argv == [
        "claude",
        "--print",
        "--output-format",
        "stream-json",
        "--verbose",
        "--permission-mode",
        "bypassPermissions",
    ]


def test_build_command_with_all_options():
    """Add model, effort, turns, tool lists, args, schema and resume."""
    inv = _build(
        _request(output_schema={"type": "object"}, resume_thread_id="sess"),
        model="m",
        effort="high",
        max_turns=5,
        allowed_tools=["Read", "Grep"],
        disallowed_tools=["Bash"],
        args=["--x"],
    )
    argv = inv.argv
    allowed = argv.index("--allowedTools")

    assert argv[argv.index("--model") + 1] == "m"
    assert argv[argv.index("--effort") + 1] == "high"
    assert argv[argv.index("--max-turns") + 1] == "5"
    assert argv[allowed + 1 : allowed + 3] == ["Read", "Grep"]
    assert argv[argv.index("--disallowedTools") + 1] == "Bash"
    assert "--x" in argv
    assert json.loads(argv[argv.index("--json-schema") + 1]) == {"type": "object"}
    assert argv[argv.index("--resume") + 1] == "sess"
    assert "--tools" not in argv


def test_build_command_respond_only_disables_tools_and_mcp():
    """Disable all tools and MCP servers for a respond-only request."""
    argv = _build(_request(allowed_actions=["respond_only"])).argv

    assert argv[argv.index("--tools") + 1] == ""
    assert "--strict-mcp-config" in argv
    assert argv[argv.index("--mcp-config") + 1] == '{"mcpServers":{}}'


def test_extract_content_serializes_structured_output_and_falls_back():
    """Use result values, structured lists, assistant text, or nothing."""
    assert _extract_content([{"type": "result", "structured_output": [1]}]) == "[1]"
    assert _extract_content([{"type": "result", "result": " hi "}]) == "hi"
    assistant = [
        {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "text", "text": "a"},
                    {"type": "tool_use"},
                    {"type": "text", "text": "b"},
                ]
            },
        },
        {"type": "user"},
    ]
    assert _extract_content(assistant) == "a\nb"
    assert _extract_content([]) == ""


def test_claude_command_passes_the_schema_natively():
    schema = load_schema("early_stop")

    command = _argv({"provider": "claude"}, _request_old(schema))

    assert json.loads(command[command.index("--json-schema") + 1]) == schema


def test_claude_command_without_schema_has_no_schema_flag():
    assert "--json-schema" not in _argv({"provider": "claude"}, _request_old({}))


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

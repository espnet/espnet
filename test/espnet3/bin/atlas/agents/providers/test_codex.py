"""Tests for codex."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from espnet3.bin.atlas.agents.cli import build_agent_config
from espnet3.bin.atlas.agents.providers import create_provider
from espnet3.bin.atlas.agents.providers.codex import (
    _collect_usage_objects,
    _format_toml_value,
)
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


def _provider(**codex):
    return create_provider(build_agent_config({"provider": "codex", "codex": codex}))


def _build(provider, request):
    return provider.build_command(
        request,
        "prompt",
        cwd=Path("/work"),
        output_path=Path("/out.txt"),
        schema_path=Path("/schema.json"),
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


def _request_old(**changes):
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


def _lines(*events):
    return "\n".join(json.dumps(event) for event in events)


def _provider_old(config):
    return create_provider(build_agent_config(config))


def test_build_command_fresh_run_with_all_options():
    """Pass sandbox, model, overrides, extra args and schema to codex exec."""
    provider = _provider(
        model="gpt-x",
        config={"model_reasoning_effort": "high", "flag": True},
        args=["--extra"],
    )

    inv = _build(provider, _request(output_schema={"type": "object"}))

    assert inv.stdin == "prompt"
    assert inv.argv[:5] == ["codex", "exec", "-C", "/work", "--json"]
    assert "--skip-git-repo-check" in inv.argv
    assert inv.argv[inv.argv.index("--sandbox") + 1] == "workspace-write"
    assert inv.argv[inv.argv.index("--model") + 1] == "gpt-x"
    assert 'model_reasoning_effort="high"' in inv.argv and "flag=true" in inv.argv
    assert "--extra" in inv.argv
    assert inv.argv[inv.argv.index("--output-schema") + 1] == "/schema.json"
    assert inv.argv[-3:] == ["--output-last-message", "/out.txt", "-"]


def test_build_command_resume_uses_sandbox_config_override():
    """Resume a thread and express the sandbox as a -c override."""
    inv = _build(_provider(), _request(resume_thread_id="th1"))

    assert inv.argv[:4] == ["codex", "exec", "resume", "th1"]
    assert "-C" not in inv.argv and "--sandbox" not in inv.argv
    assert 'sandbox_mode="workspace-write"' in inv.argv
    assert "--output-schema" not in inv.argv


def test_build_command_resume_full_access_bypasses_sandbox():
    """Use the bypass flag when resuming with full access."""
    inv = _build(
        _provider(sandbox="danger-full-access", skip_git_repo_check=False),
        _request(resume_thread_id="th1"),
    )

    assert "--dangerously-bypass-approvals-and-sandbox" in inv.argv
    assert "--skip-git-repo-check" not in inv.argv


def test_build_command_without_sandbox_omits_sandbox_options():
    """Emit no sandbox option when the sandbox is unset."""
    inv = _build(_provider(sandbox=None), _request())
    assert "--sandbox" not in inv.argv


@pytest.mark.parametrize(
    "value, text",
    [
        (True, "true"),
        (False, "false"),
        (3, "3"),
        (1.5, "1.5"),
        ('a"b', '"a\\"b"'),
        ([1, "x"], '[1, "x"]'),
        ({"k": [True]}, "{k = [true]}"),
    ],
)
def test_format_toml_value_serializes_supported_types(value, text):
    """Serialize scalars, lists and tables for -c overrides."""
    assert _format_toml_value(value) == text


@pytest.mark.parametrize("value, exc", [(None, ValueError), (object(), TypeError)])
def test_format_toml_value_rejects_unsupported_values(value, exc):
    """Reject null and unknown types."""
    with pytest.raises(exc):
        _format_toml_value(value)


def test_collect_usage_objects_finds_nested_usage_and_token_fields():
    """Gather usage dicts and bare token counters from nested events."""
    event = {
        "a": [{"usage": {"input_tokens": 1}}],
        "b": {"output_tokens": 2, "total_tokens": "x", "other": 1},
    }

    found = _collect_usage_objects(event)

    assert {"input_tokens": 1} in found
    assert {"output_tokens": 2} in found
    assert _collect_usage_objects("scalar") == []


def test_build_env_creates_workspace_home_and_copies_credentials(tmp_path, monkeypatch):
    """Copy auth files once into a recipe-local CODEX_HOME."""
    user_home = tmp_path / "user"
    user_home.mkdir()
    (user_home / "auth.json").write_text("secret")
    monkeypatch.setenv("CODEX_HOME", str(user_home))
    recipe = tmp_path / "recipe"
    recipe.mkdir()

    env = _provider().build_env(recipe)

    assert env["CODEX_HOME"] == str(recipe / ".codex")
    assert (recipe / ".codex" / "auth.json").read_text() == "secret"
    (user_home / "auth.json").write_text("new")
    _provider().build_env(recipe)
    assert (recipe / ".codex" / "auth.json").read_text() == "secret"


def test_parse_without_any_message_returns_empty_content(tmp_path):
    """Return empty content and no thread for a silent run."""
    parsed = _provider().parse("not json\n", tmp_path / "missing.txt")
    assert (parsed.content, parsed.thread_id) == ("", None)


def test_record_usage_appends_history_and_handles_no_usage(tmp_path):
    """Append one history line per call even without usage events."""
    provider = _provider()

    provider.record_usage(tmp_path, _request(), ["codex"], "")
    provider.record_usage(tmp_path, _request(), ["codex"], "")

    summary = json.loads((tmp_path / "agent_usage.json").read_text())
    assert summary["last_usage"] == {} and summary["event_count"] == 0
    lines = (tmp_path / "agent_usage_history.jsonl").read_text().splitlines()
    assert len(lines) == 2


def test_codex_command_passes_nested_config_overrides_as_toml():
    command = _argv(
        {
            "provider": "codex",
            "codex": {"config": {"model_provider": "vllm", "retries": [1, 2]}},
        },
        _request(),
    )

    assert 'model_provider="vllm"' in command
    assert "retries=[1, 2]" in command


def test_codex_command_constrains_the_final_reply_to_the_schema():
    request = _request(output_schema=load_schema("early_stop"))
    command = _argv({"provider": "codex"}, request)

    assert command[command.index("--output-schema") + 1] == "schema.json"
    assert "--output-schema" not in _argv({"provider": "codex"}, _request())


def test_codex_provider_falls_back_to_the_last_agent_message(tmp_path):
    output = tmp_path / "out.txt"
    output.write_text("")
    provider = _provider_old({"provider": "codex"})
    stdout = _lines(
        {"type": "thread.started", "thread_id": "th1"},
        {"type": "item.completed", "item": {"type": "agent_message", "text": " hi "}},
    )

    parsed = provider.parse(stdout, output)

    assert (parsed.content, parsed.thread_id) == ("hi", "th1")
    output.write_text("from file\n")
    assert provider.parse(stdout, output).content == "from file"


def test_codex_provider_records_token_usage(tmp_path):
    provider = _provider_old({"provider": "codex"})

    provider.record_usage(
        tmp_path,
        _request_old(),
        ["codex", "exec"],
        _lines({"type": "turn.completed", "usage": {"input_tokens": 3}}),
    )

    summary = json.loads((tmp_path / "agent_usage.json").read_text())
    assert summary["last_usage"] == {"input_tokens": 3}
    assert (tmp_path / "agent_usage_history.jsonl").is_file()

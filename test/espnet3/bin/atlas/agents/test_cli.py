"""Tests for cli."""

from __future__ import annotations

import stat
import sys

import pytest
from omegaconf import OmegaConf

from espnet3.bin.atlas.agents.cli import CliAgent, build_agent_config, render_prompt
from espnet3.bin.atlas.core.models import AgentRequest
from espnet3.bin.atlas.resources import load_schema


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


def test_validate_output_accepts_a_conforming_reply():
    CliAgent._validate_output(
        {"should_stop": False, "reason": "still improving", "key_metrics": []},
        load_schema("early_stop"),
    )


@pytest.mark.parametrize(
    "reply, fragment",
    [
        ({"should_stop": "no", "reason": "r", "key_metrics": []}, "should_stop"),
        ({"should_stop": True, "reason": "", "key_metrics": []}, "reason"),
        ({"should_stop": True, "reason": "r"}, "key_metrics"),
    ],
)
def test_validate_output_names_the_failing_field(reply, fragment):
    with pytest.raises(ValueError, match=fragment):
        CliAgent._validate_output(reply, load_schema("early_stop"))


def test_propose_schema_has_no_top_level_combinators():
    # The Claude API rejects oneOf/allOf/anyOf at the top level of a schema.
    schema = load_schema("propose")

    assert not {"oneOf", "allOf", "anyOf"} & set(schema)
    with pytest.raises(ValueError, match="config_patch"):
        CliAgent._validate_output(
            {
                "rationale": "r",
                "config_patch": [{"key": "optimizer.lr", "value": 0.1}],
                "expected_effect": "e",
                "risk": "k",
                "extra_stages": [],
            },
            schema,
        )


def test_validate_output_without_a_schema_accepts_anything():
    CliAgent._validate_output({"anything": 1}, {})


def test_build_agent_config_fills_defaults_and_keeps_overrides():
    config = build_agent_config(
        {"provider": "codex", "codex": {"model": "local", "args": ["--full-auto"]}}
    )

    assert config.codex.model == "local"
    assert config.codex.sandbox == "workspace-write"
    assert config.codex.args == ["--full-auto"]
    assert config.claude.permission_mode == "bypassPermissions"


def test_unknown_provider_is_rejected():
    with pytest.raises(ValueError, match="provider must be one of"):
        build_agent_config({"provider": "vibe"})


def test_render_prompt_shows_the_schema():
    prompt = render_prompt(_request(output_schema={"type": "object"}))

    assert prompt.startswith("Task: t\n\nFollow the requested output shape exactly.")
    assert '"type": "object"' in prompt


def test_cli_agent_returns_the_schema_constrained_reply(tmp_path):
    from espnet3.bin.atlas.resources import load_schema

    reply = {"should_stop": False, "reason": "ok", "key_metrics": []}
    agent = CliAgent(
        {
            "provider": "claude",
            "claude": {"executable": str(_fake_claude(tmp_path, reply))},
        },
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
        {
            "provider": "claude",
            "claude": {"executable": str(_fake_claude(tmp_path, reply))},
        },
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
    with pytest.raises(ValueError, match="provider must be one of 'codex', 'claude'"):
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

"""Tests for CliAgent's structured-output handling."""

from __future__ import annotations

import json

import pytest

from espnet3.bin.atlas.agents.cli import CliAgent
from espnet3.bin.atlas.core.models import AgentConfig, AgentRequest
from espnet3.bin.atlas.resources import load_schema


def _request(schema):
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


def test_claude_command_passes_the_schema_natively():
    agent = object.__new__(CliAgent)
    agent.config = AgentConfig(provider="claude")
    schema = load_schema("early_stop")

    command = agent._build_claude_command(_request(schema))

    assert json.loads(command[command.index("--json-schema") + 1]) == schema


def test_claude_command_without_schema_has_no_schema_flag():
    agent = object.__new__(CliAgent)
    agent.config = AgentConfig(provider="claude")

    assert "--json-schema" not in agent._build_claude_command(_request({}))


def test_validate_output_accepts_a_conforming_reply():
    CliAgent._validate_output(
        {"should_stop": False, "reason": "still improving", "key_metrics": {}},
        load_schema("early_stop"),
    )


@pytest.mark.parametrize(
    "reply, fragment",
    [
        ({"should_stop": "no", "reason": "r", "key_metrics": {}}, "should_stop"),
        ({"should_stop": True, "reason": "", "key_metrics": {}}, "reason"),
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
    with pytest.raises(ValueError, match="config_patches"):
        CliAgent._validate_output(
            {
                "rationale": "r",
                "config_patch": {"optimizer.lr": 0.1},
                "expected_effect": "e",
                "risk": "k",
                "extra_stages": [],
            },
            schema,
        )


def test_validate_output_without_a_schema_accepts_anything():
    CliAgent._validate_output({"anything": 1}, {})

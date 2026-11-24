"""Tests for the packaged Atlas prompt templates."""

from __future__ import annotations

import re
from importlib import resources

import pytest

from espnet3.bin.atlas.resources import load_prompt, load_schema

_PROMPTS = resources.files("espnet3.bin.atlas").joinpath("resources", "prompts")


def test_load_prompt_substitutes_every_placeholder_without_rescanning_values():
    text = load_prompt("job_operate", instructions="bad ${thing} {x}")

    assert text.endswith("bad ${thing} {x}")
    assert not text.endswith("\n")


def test_load_prompt_rejects_missing_values():
    with pytest.raises(KeyError):
        load_prompt("job_operate")


def test_every_template_is_loadable_with_exactly_its_placeholders():
    for path in _PROMPTS.iterdir():
        if not path.name.endswith(".md"):
            continue
        names = set(re.findall(r"\$\{(\w+)\}", path.read_text(encoding="utf-8")))
        text = load_prompt(path.name[:-3], **{name: f"<{name}>" for name in names})
        assert "${" not in text, path.name


def test_missing_schema_raises():
    with pytest.raises(FileNotFoundError):
        load_schema("does_not_exist")


def test_every_packaged_schema_is_a_valid_json_schema():
    from jsonschema import Draft202012Validator

    names = [
        path.name[:-5]
        for path in resources.files("espnet3.bin.atlas")
        .joinpath("resources", "schemas")
        .iterdir()
        if path.name.endswith(".json")
    ]
    assert {"job_submit", "job_status", "early_stop"} <= set(names)
    for name in names:
        Draft202012Validator.check_schema(load_schema(name))


def test_every_packaged_schema_is_strict():
    """Constrained decoding needs closed objects with every property required."""

    def check(node, where):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node.get("additionalProperties") is False, where
                assert set(node["required"]) == set(node["properties"]), where
            for key, value in node.items():
                check(value, f"{where}.{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                check(value, f"{where}[{index}]")

    for path in (
        resources.files("espnet3.bin.atlas").joinpath("resources", "schemas").iterdir()
    ):
        if path.name.endswith(".json"):
            check(load_schema(path.name[:-5]), path.name)

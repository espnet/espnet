"""Prompt templates, JSON schemas and defaults shipped with Atlas, and their loaders."""

from __future__ import annotations

import json
from importlib.resources import files
from string import Template
from typing import Any

from omegaconf import DictConfig, OmegaConf

_ROOT = files(__package__)


def load_prompt(name: str, **values: Any) -> str:
    """Render the prompt template ``resources/prompts/<name>.md``.

    Placeholders are written ``${name}``. Every placeholder must be given a
    value; unknown or missing names raise, so a typo cannot silently produce a
    prompt with a literal ``${...}`` in it. Values are inserted as is and are
    not scanned for further placeholders.

    Args:
        name: Template name without the ``.md`` suffix.
        **values: Values for the template's placeholders.

    Returns:
        str: The rendered prompt, without the template file's final newline.

    Raises:
        KeyError: A placeholder in the template has no value.
        FileNotFoundError: There is no template with this name.

    Example:
        >>> load_prompt("json_repair", error="missing field").startswith("Your")
        True
    """
    text = _ROOT.joinpath("prompts", f"{name}.md").read_text(encoding="utf-8")
    return Template(text.removesuffix("\n")).substitute(
        {key: str(value) for key, value in values.items()}
    )


def load_schema(name: str) -> dict[str, Any]:
    """Read the JSON file ``resources/schemas/<name>.json``.

    Args:
        name: Schema name without the ``.json`` suffix.

    Returns:
        dict[str, Any]: The parsed JSON document.

    Example:
        >>> load_schema("job_status")["type"]
        'object'
    """
    text = _ROOT.joinpath("schemas", f"{name}.json").read_text(encoding="utf-8")
    return json.loads(text)


def convert_shape_to_schema(shape: dict[str, Any]) -> dict[str, Any]:
    """Turn a graph node's ``output_schema`` example shape into a JSON Schema.

    Recipe graphs describe a skill node's extra reply fields as an example
    object whose scalar values are only descriptions, e.g.
    ``{"status": "continue or done", "payload": {}}``. Every root field is
    required, a ``{}`` or ``[]`` value requires a JSON object or array, and
    scalar values are described but not type-checked.

    Args:
        shape: The example object.

    Returns:
        dict[str, Any]: The equivalent JSON Schema.

    Example:
        >>> convert_shape_to_schema({"summary": "one line", "files": []})["required"]
        ['summary', 'files']
    """

    def convert(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return {
                "type": "object",
                "properties": {key: convert(item) for key, item in value.items()},
                "required": [
                    key for key, item in value.items() if isinstance(item, (dict, list))
                ],
            }
        if isinstance(value, list):
            return {"type": "array"}
        return {"description": str(value)}

    schema = convert(shape)
    schema["required"] = list(shape)
    return schema


def load_atlas_defaults() -> DictConfig:
    """Read ``resources/atlas_defaults.yaml``, the defaults of a study config.

    Returns:
        DictConfig: The default study settings, under the ``atlas`` key.

    Example:
        >>> load_atlas_defaults().atlas.tick_interval_sec_idle
        60
    """
    text = _ROOT.joinpath("atlas_defaults.yaml").read_text(encoding="utf-8")
    return OmegaConf.create(text)


def load_agent_defaults() -> DictConfig:
    """Read ``resources/agent_defaults.yaml``, the defaults of ``atlas.agent``.

    Returns:
        DictConfig: The default coding-agent settings.

    Example:
        >>> load_agent_defaults().codex.sandbox
        'workspace-write'
    """
    text = _ROOT.joinpath("agent_defaults.yaml").read_text(encoding="utf-8")
    return OmegaConf.create(text)

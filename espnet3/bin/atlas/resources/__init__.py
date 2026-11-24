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
        >>> load_prompt("job_operate", instructions="qsub").startswith("Operate")
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

"""Graph loading and routing."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from omegaconf import OmegaConf

from espnet3.bin.atlas.core import AtlasError
from espnet3.bin.atlas.core.models import NodeConfig, ResearchGraph, Resources


def load_stage_class(target: str, recipe_dir: Path | None = None) -> type:
    """Import a stage class from its dotted path.

    Args:
        target: Module and class path, such as ``package.module.Stage``.
        recipe_dir: Optional recipe root used to resolve recipe-local modules.

    Returns:
        type: Imported stage class.

    Example:
        >>> load_stage_class("espnet3.bin.atlas.stages.skill.SkillStage")
        <class 'espnet3.bin.atlas.stages.skill.SkillStage'>
    """
    if recipe_dir is not None:
        recipe_dir_str = str(recipe_dir.resolve())
        if recipe_dir_str not in sys.path:
            sys.path.insert(0, recipe_dir_str)
    module_path, class_name = target.rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        hint = (
            f"\nIf this is recipe-local, verify recipe_dir='{recipe_dir}' and "
            f"module file '{recipe_dir / module_path.replace('.', '/')}.py'."
            if recipe_dir is not None
            else ""
        )
        raise AtlasError(
            f"Cannot import module '{module_path}' for stage target '{target}'."
            f"{hint}\nOriginal error: {exc}"
        ) from exc
    if not hasattr(module, class_name):
        raise AtlasError(f"Module '{module_path}' has no class '{class_name}'.")
    return getattr(module, class_name)


def load_graph(path: Path, recipe_dir: Path | None = None) -> ResearchGraph:
    """Load and validate a graph YAML, including its stage imports.

    Args:
        path: YAML file containing the graph declaration.
        recipe_dir: Optional recipe root for recipe-local stage classes.

    Returns:
        ResearchGraph: Validated graph with resolved node configuration.

    Example:
        >>> graph = load_graph(Path("graph.yaml"))
        >>> graph.start
        'research_context'
    """
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, dict):
        raise AtlasError(f"Graph YAML must be a mapping: {path}")
    nodes_raw = raw.get("nodes")
    if not nodes_raw:
        raise AtlasError("Graph must define non-empty `nodes`.")
    nodes: dict[str, NodeConfig] = {}
    for name, cfg in nodes_raw.items():
        skill = str(cfg["skill"]) if cfg.get("skill") else None
        target = str(cfg.get("target") or "")
        if not target:
            raise AtlasError(f"Node '{name}' needs a `target` stage class.")
        load_stage_class(target, recipe_dir=recipe_dir)
        execution = str(cfg.get("execution", "local"))
        if execution not in {"local", "submission"}:
            raise AtlasError(
                f"Node '{name}' execution must be 'local' or 'submission', "
                f"got {execution!r}."
            )
        resources_cfg = cfg.get("resources")
        resources = Resources.from_config(resources_cfg) if resources_cfg else None
        on = cfg.get("on", {})
        terminal = bool(cfg.get("terminal", False))
        output_schema = dict(cfg.get("output_schema", {}) or {})
        for field_name, field_schema in output_schema.items():
            if not isinstance(field_schema, dict) or "type" not in field_schema:
                raise AtlasError(
                    f"Node '{name}' output_schema.{field_name} must be a JSON Schema "
                    "with a `type`, e.g. {type: string, description: ...}."
                )
        for key in (cfg.get("output_key"), *(cfg.get("trial_update_keys") or [])):
            if key and key not in output_schema:
                raise AtlasError(
                    f"Node '{name}' uses '{key}' but does not declare it in "
                    "output_schema."
                )
        if "status" in output_schema or "message" in output_schema:
            raise AtlasError(
                f"Node '{name}' output_schema must not declare status or message; "
                "they are added for every skill node."
            )
        nodes[name] = NodeConfig(
            name=name,
            target=target,
            execution=execution,
            resources=resources,
            skill=skill,
            skill_context=dict(cfg.get("skill_context", {}) or {}),
            output_schema=output_schema,
            allowed_actions=[
                str(action) for action in cfg.get("allowed_actions", []) or []
            ],
            trial_update_keys=[
                str(key) for key in cfg.get("trial_update_keys", []) or []
            ],
            output_file=str(cfg["output_file"]) if cfg.get("output_file") else None,
            output_key=str(cfg["output_key"]) if cfg.get("output_key") else None,
            output_append=bool(cfg.get("output_append", False)),
            fallback_output=(
                str(cfg["fallback_output"]) if cfg.get("fallback_output") else None
            ),
            on=on,
            terminal=terminal,
        )
    start = raw.get("start")
    if start not in nodes:
        raise AtlasError(f"Graph start node '{start}' is not defined.")
    graph_name = str(raw.get("name", path.stem))
    version = int(raw.get("version", 1))
    return ResearchGraph(name=graph_name, version=version, start=start, nodes=nodes)

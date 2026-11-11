"""Config helpers for Atlas hyperparameter search."""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from espnet3.bin.atlas.core.serialization import save_yaml
from espnet3.utils.config_utils import load_and_merge_config


def load_atlas_config(config_path: Path):
    """Load the recipe atlas config merged with TEMPLATE defaults."""
    return load_and_merge_config(config_path, config_name="atlas.yaml", resolve=True)


def load_recipe_stage_config(
    recipe_dir: Path, config_path: Path, config_name: str, resolve: bool = True
):
    """Load a recipe config merged with TEMPLATE defaults."""
    return load_and_merge_config(
        recipe_dir / config_path,
        config_name=config_name,
        resolve=resolve,
    )


def write_resolved_config(path: Path, config) -> None:
    """Write a resolved config to disk."""
    save_yaml(path, OmegaConf.to_container(config, resolve=True))

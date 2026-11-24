"""Config helpers for Atlas hyperparameter search."""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from espnet3.bin.atlas.core.serialization import save_yaml
from espnet3.utils.config_utils import load_and_merge_config


def load_atlas_config(config_path: Path):
    """Load a recipe's ``atlas.yaml``, merged with its TEMPLATE defaults.

    The default package (``egs3.TEMPLATE.<system>``) is inferred from
    ``config_path`` living under ``egs3/<corpus>/<system>/conf/``, the same
    way other recipe configs (``training.yaml``, etc.) are merged.

    Args:
        config_path: Path to the recipe's ``atlas.yaml``.

    Returns:
        DictConfig: The merged, resolved config, with the study's settings
        under its ``atlas`` key.

    Example:
        >>> config = load_atlas_config(Path("egs3/mini_an4/esp2_asr/conf/atlas.yaml"))
        >>> config.atlas.study_name
        'mini_an4_ci_check'
    """
    return load_and_merge_config(config_path, config_name="atlas.yaml", resolve=True)


def load_recipe_stage_config(
    recipe_dir: Path, config_path: Path, config_name: str, resolve: bool = True
):
    """Load one of a recipe's stage configs, merged with its TEMPLATE defaults.

    Used by Atlas stages to read a trial's ``training_config``/
    ``inference_config``/``metrics_config`` the same way ``run.py`` does.

    Args:
        recipe_dir: Recipe root that ``config_path`` is relative to.
        config_path: Stage config path, relative to ``recipe_dir`` (e.g.
            ``conf/training.yaml``).
        config_name: Config filename used to locate the TEMPLATE default,
            e.g. ``"training.yaml"``.
        resolve: Whether to resolve OmegaConf interpolations in the merged
            result.

    Returns:
        DictConfig: The merged config for that stage.

    Example:
        >>> config = load_recipe_stage_config(
        ...     Path("egs3/mini_an4/esp2_asr"), Path("conf/training.yaml"), "training.yaml"
        ... )
    """
    return load_and_merge_config(
        recipe_dir / config_path,
        config_name=config_name,
        resolve=resolve,
    )


def write_resolved_config(path: Path, config) -> None:
    """Write a fully resolved config to disk as plain YAML.

    Used to persist a study's merged ``atlas.yaml`` into its study directory
    so later commands (``run``/``status``/``stop``) can load it with a plain
    ``OmegaConf.load`` and still see every TEMPLATE-merged default.

    Args:
        path: Destination YAML file path.
        config: OmegaConf config to resolve and write.

    Returns:
        None.

    Example:
        >>> write_resolved_config(Path("exp/atlas/my_study/atlas.yaml"), config)
    """
    save_yaml(path, OmegaConf.to_container(config, resolve=True))

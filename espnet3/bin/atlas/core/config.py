"""Config helpers for Atlas hyperparameter search."""

from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path

from omegaconf import OmegaConf

from espnet3.bin.atlas.core.serialization import save_yaml
from espnet3.bin.atlas.resources import load_atlas_defaults
from espnet3.utils.config_utils import load_and_merge_config


def load_atlas_config(config_path: Path):
    """Load a study config (a recipe's ``conf/atlas*.yaml``) over the Atlas defaults.

    The defaults are ``resources/atlas_defaults.yaml``; they do not depend on the
    recipe's system or location. ``atlas.submission`` has no default, since how
    jobs are submitted depends on the cluster, and must be given.

    Args:
        config_path: Path to the study config.

    Returns:
        DictConfig: The merged, resolved config, with the study's settings
        under its ``atlas`` key.

    Raises:
        ValueError: If ``atlas.submission`` is missing, ``controller.run_in`` is
            not ``local`` or ``queue``, or ``queue`` is given without
            ``atlas.submission.instructions``.

    Example:
        >>> config = load_atlas_config(Path("egs3/mini_an4/esp2_asr/conf/atlas.yaml"))
        >>> config.atlas.study_name
        'mini_an4_ci_check'
    """
    config = OmegaConf.merge(load_atlas_defaults(), OmegaConf.load(config_path))
    OmegaConf.resolve(config)
    submission = config.atlas.get("submission")
    if submission is None:
        raise ValueError(
            "atlas.submission is required: give atlas.submission.instructions "
            "(how to submit jobs) and atlas.submission.resources; leave the "
            "instructions out to run jobs as local processes"
        )
    run_in = submission.get("controller", {}).get("run_in", "local")
    if run_in not in ("local", "queue"):
        raise ValueError(
            f"atlas.submission.controller.run_in must be 'local' or 'queue', "
            f"got {run_in!r}"
        )
    if run_in == "queue" and not submission.get("instructions"):
        raise ValueError(
            "atlas.submission.instructions is required when "
            "atlas.submission.controller.run_in is 'queue'"
        )
    return config


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


def split_patch_by_search_space(
    patch: dict, search_space
) -> tuple[dict, list[str]]:
    """Split a config patch into allowed keys and keys the search space rules out.

    A key is dropped when it matches any ``denied_keys`` pattern, or when
    ``allowed_keys`` is set and it matches none of its patterns. Patterns are
    ``fnmatch`` globs over the dotted key, for example ``optimizer.*``.

    Args:
        patch: Dotted config keys mapped to new values.
        search_space: The ``atlas.search_space`` config node (``allowed_keys`` and
            ``denied_keys`` are both optional).

    Returns:
        tuple[dict, list[str]]: The allowed part of ``patch`` and the removed keys.

    Example:
        >>> space = {"allowed_keys": ["optimizer.*"], "denied_keys": []}
        >>> split_patch_by_search_space({"optimizer.lr": 1, "trainer.x": 2}, space)
        ({'optimizer.lr': 1}, ['trainer.x'])
    """
    allowed = list(getattr(search_space, "allowed_keys", []) or [])
    denied = list(getattr(search_space, "denied_keys", []) or [])
    kept: dict = {}
    removed: list[str] = []
    for key, value in patch.items():
        key = str(key)
        if any(fnmatch(key, p) for p in denied) or (
            allowed and not any(fnmatch(key, p) for p in allowed)
        ):
            removed.append(key)
        else:
            kept[key] = value
    return kept, removed

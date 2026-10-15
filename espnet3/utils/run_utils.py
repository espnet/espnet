"""Shared, role-agnostic config helpers for espnet3 runner entry points."""

from __future__ import annotations

from typing import Mapping

from omegaconf import DictConfig, OmegaConf


class ConfigError(ValueError):
    """A config is missing what a stage needs, or fails to resolve.

    Examples:
        >>> raise ConfigError("train config needs model: block")
        Traceback (most recent call last):
        espnet3.utils.run_utils.ConfigError: train config needs model: block
    """


def resolve_loaded_configs(configs: Mapping[str, DictConfig]) -> None:
    """Resolve each config in `configs` in place, naming its key on failure.

    Runner entry points load configs with `resolve=False` so the inherited,
    merged config can be assembled from several stages' top-level keys
    before any `${...}` interpolation is evaluated; this is the final
    resolution step, once that assembly is done.

    Args:
        configs (Mapping[str, DictConfig]): Mapping from a name - a stage
            name, in this codebase - to its loaded, not-yet-resolved config.

    Returns:
        None: Each config is resolved in place.

    Raises:
        ConfigError: An interpolation in one of the configs cannot be
            resolved; the message names the mapping key and the original
            error.

    Examples:
        >>> from omegaconf import OmegaConf
        >>> configs = {
        ...     "train": OmegaConf.create(
        ...         {"exp_tag": "t", "exp_dir": "./exp/${exp_tag}"}
        ...     )
        ... }
        >>> resolve_loaded_configs(configs)
        >>> configs["train"].exp_dir
        './exp/t'
    """
    for key, config in configs.items():
        try:
            OmegaConf.resolve(config)
        except Exception as e:
            raise ConfigError(
                f"{key} config cannot resolve an interpolation: {e}"
            ) from e

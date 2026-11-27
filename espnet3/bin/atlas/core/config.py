"""Config helpers for Atlas hyperparameter search."""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from espnet3.bin.atlas.resources import load_atlas_defaults


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

"""Generic entry point: parse CLI args and run a system's declared stages."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from espnet3.components.contract.stages import (
    check_requested_stages,
    roles,
    stage_names,
)
from espnet3.utils.config_utils import load_and_merge_config
from espnet3.utils.logging_utils import configure_logging
from espnet3.utils.stages_utils import resolve_stages, run_stages


def build_parser(system_cls: type) -> argparse.ArgumentParser:
    """Build the CLI parser for ``system_cls``: stages and per-role configs.

    Args:
        system_cls: The system class whose declared stages set the
            ``--stages`` choices.

    Returns:
        A parser with ``--stages``, one ``--<role>_config`` per
        :func:`~espnet3.components.contract.stages.roles` of
        ``system_cls``, ``--exp_dir``, ``--dry_run``, and
        ``--write_requirements``.

    Examples:
        >>> from espnet3.systems.esp2_asr.system import ASRSystem
        >>> parser = build_parser(ASRSystem)
        >>> args = parser.parse_args(["--stages", "train"])
        >>> args.stages
        ['train']
    """
    names = stage_names(system_cls)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stages",
        choices=names + ["all"],
        nargs="+",
        default=names,
        help="Which stages to run. Multiple values allowed.",
    )
    for role in roles(system_cls):
        parser.add_argument(
            f"--{role}_config",
            default=None,
            type=Path,
            help=f"Hydra config for the {role} role's stages.",
        )
    parser.add_argument(
        "--exp_dir",
        default=None,
        type=str,
        help="Experiment directory to bake stage configs under and read "
        "earlier baked configs from. Required for a standalone run whose "
        "own config has no exp_dir of its own.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be executed without actually running stages.",
    )
    parser.add_argument(
        "--write_requirements",
        action="store_true",
        help="Write requirements.txt alongside each stage log.",
    )
    return parser


def default_conf_package(system_cls: type) -> str:
    """Return the ``egs3.TEMPLATE`` package that holds ``system_cls``'s default configs.

    Derived from ``system_cls``'s own module, ``espnet3.systems.<x>.system``,
    by the same ``<x>`` naming :data:`~espnet3.api.inference.loading.SYSTEM_ALIASES`
    uses elsewhere.

    Examples:
        >>> from espnet3.systems.esp2_asr.system import ASRSystem
        >>> default_conf_package(ASRSystem)
        'egs3.TEMPLATE.esp2_asr'
    """
    system_name = system_cls.__module__.split(".")[-2]
    return f"egs3.TEMPLATE.{system_name}"


def launch(
    system_cls: type,
    *,
    conf_package: str | None = None,
    argv: Sequence[str] | None = None,
    parser: argparse.ArgumentParser | None = None,
) -> None:
    """Parse CLI args, build ``system_cls``, and run its requested stages.

    Loads each config role's own config (the default plus whatever the
    caller gave, not yet resolved), passes them to ``system_cls`` to build
    each requested stage's merged, resolved config (see
    :class:`~espnet3.systems.base.system.BaseSystem`), then runs the
    requested stages, baking each stage's own config before it runs.

    Args:
        system_cls: The system class to instantiate and run.
        conf_package: Package holding the default configs; defaults to
            :func:`default_conf_package`.
        argv: Arguments to parse instead of ``sys.argv``.
        parser: A parser to use instead of ``build_parser(system_cls)``,
            for a recipe that needs its own extra arguments.

    Raises:
        espnet3.components.contract.stages.StageContractError: A requested
            stage is not declared.
        RuntimeError: No ``--exp_dir`` is given and no role's own config
            has a self-resolving ``exp_dir``.

    Examples:
        ```python
        # egs3/mini_an4/esp2_asr/run.py
        from espnet3.systems.esp2_asr.system import ASRSystem
        from espnet3.systems.launch import launch

        if __name__ == "__main__":
            launch(ASRSystem)
        ```

        The ``__main__`` guard is required: ``collect_stats``/``train``
        spawn Dask worker processes that re-import ``run.py``; without
        the guard, each worker re-runs ``launch(...)`` itself.
    """
    parser = parser or build_parser(system_cls)
    args = parser.parse_args(argv)
    names = stage_names(system_cls)
    stages_to_run = resolve_stages(args.stages, names)
    system_roles = roles(system_cls)
    cli_paths = {role: getattr(args, f"{role}_config") for role in system_roles}
    check_requested_stages(
        system_cls, stages_to_run, provided=cli_paths, exp_dir=args.exp_dir
    )

    package = conf_package or default_conf_package(system_cls)
    configs = {
        role: (
            load_and_merge_config(
                cli_paths[role],
                config_name=f"{role}.yaml",
                default_package=package,
                resolve=False,
            )
            if cli_paths[role] is not None
            else None
        )
        for role in system_roles
    }

    logger = configure_logging()
    system = system_cls(
        configs=configs, exp_dir=args.exp_dir, stages_to_run=stages_to_run
    )

    logger.info("System: %s", system_cls.__name__)
    logger.info("Requested stages: %s", args.stages)
    logger.info("Resolved stages: %s", stages_to_run)

    run_stages(system=system, stages_to_run=stages_to_run, args=args, log=logger)

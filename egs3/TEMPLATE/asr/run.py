#!/usr/bin/env python3
"""Generic runner template for System-based experiments."""

from __future__ import annotations

import argparse
import logging
import sys
from importlib import resources
from pathlib import Path
from typing import Any, Callable, List, Sequence, Tuple, Type

from omegaconf import OmegaConf

from espnet3.utils.config_utils import load_config_with_defaults
from espnet3.utils.logging_utils import (
    configure_logging,
    log_env_metadata,
    log_run_metadata,
)
from espnet3.utils.stages_utils import resolve_stages, run_stages

# Default stage list (can be extended/overridden by callers)
DEFAULT_STAGES: List[str] = [
    "create_dataset",
    "train_tokenizer",
    "collect_stats",
    "train",
    "infer",
    "measure",
]

# Type alias for a System class
SystemCls = Type[Any]
AddArgsFn = Callable[[argparse.ArgumentParser], None]
logger = logging.getLogger(__name__)


def build_parser(
    stages: Sequence[str],
) -> argparse.ArgumentParser:
    """Build base ArgumentParser and let caller extend it."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stages",
        choices=list(stages) + ["all"],
        nargs="+",
        default=["all"],
        help="Which stages to run. Multiple values allowed.",
    )
    parser.add_argument(
        "--train_config",
        default=None,
        type=Path,
        help=(
            "Hydra config for training (passed to load_config_with_defaults). "
            "Required for create_dataset/train_tokenizer/collect_stats/train stages."
        ),
    )
    parser.add_argument(
        "--infer_config",
        default=None,
        type=Path,
        help="Hydra config for infer stage.",
    )
    parser.add_argument(
        "--measure_config",
        default=None,
        type=Path,
        help="Hydra config for measure stage.",
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


def parse_cli_and_stage_args(
    parser: argparse.ArgumentParser,
    stages: Sequence[str],
) -> Tuple[argparse.Namespace, List[str]]:
    args = parser.parse_args()
    stages_to_run = resolve_stages(args.stages, stages)

    return args, stages_to_run


def _resolve_template_config_filename(config_arg_name: str) -> str:
    """Resolve template YAML filename from a CLI config argument key.

    Args:
        config_arg_name: Internal config argument key name
            (``train_config``, ``infer_config``, or ``measure_config``).

    Returns:
        str: Template config filename under ``egs3.TEMPLATE.asr/conf``.

    Raises:
        ValueError: If an unknown config argument name is provided.

    Examples:
        >>> _resolve_template_config_filename("train_config")
        'training.yaml'
        >>> _resolve_template_config_filename("infer_config")
        'inference.yaml'

    Todo:
        If new config are added to CLI options, extend this mapping.
    """
    if config_arg_name == "train_config":
        return "training.yaml"
    if config_arg_name == "infer_config":
        return "inference.yaml"
    if config_arg_name == "measure_config":
        return "metrics.yaml"
    raise ValueError(f"Unknown config argument name: {config_arg_name}")


def _load_template_defaults(config_arg_name: str):
    """Load packaged TEMPLATE defaults by config kind.

    Args:
        config_arg_name: Internal config argument key name
            (``train_config``, ``infer_config``, or ``measure_config``).

    Returns:
        DictConfig: Loaded template default config.
        
    Examples:
        >>> cfg = _load_template_defaults("train_config")
        >>> "dataset" in cfg
        True
        >>> cfg = _load_template_defaults("infer_config")
        >>> cfg.get("provider", {}).get("_target_") is not None
        True

    Note:
        Uses ``importlib.resources`` to avoid hard-coded file system paths, so
        it also works after ``pip install`` as package data.

    """
    filename = _resolve_template_config_filename(config_arg_name)
    package = "egs3.TEMPLATE.asr"
    resource = resources.files(package).joinpath("conf", filename)
    with resources.as_file(resource) as path:
        return load_config_with_defaults(str(path))
    return OmegaConf.create({})


def _load_and_merge_config(config_path: Path | None, config_arg_name: str):
    """Load user config and merge with TEMPLATE defaults.

    Args:
        config_path: Path to the user-provided YAML config. If ``None``,
            this function returns ``None``.
        config_arg_name: Internal config argument key name used to select
            the corresponding TEMPLATE default config.

    Returns:
        DictConfig | None: Merged config if ``config_path`` is provided,
        otherwise ``None``.

    Examples:
        >>> cfg = _load_and_merge_config(
        ...     Path("egs3/mini_an4/asr/conf/inference.yaml"),
        ...     "infer_config",
        ... )
        >>> cfg.provider._target_
        'espnet3.systems.base.inference_provider.InferenceProvider'
        >>> _load_and_merge_config(None, "measure_config") is None
        True

    Note:
        Merge order is ``template_defaults -> user_config``. User config wins
        on key conflicts, so recipe-specific overrides remain explicit.

    """
    if config_path is None:
        return None
    user_cfg = load_config_with_defaults(config_path)
    default_cfg = _load_template_defaults(config_arg_name)
    return OmegaConf.merge(default_cfg, user_cfg)


def main(
    args,
    system_cls: SystemCls,
    stages: Sequence[str] = DEFAULT_STAGES,
) -> None:
    stages_to_run = resolve_stages(args.stages, stages)

    # -----------------------------------------
    # Load configs
    # -----------------------------------------
    train_config = _load_and_merge_config(args.train_config, "train_config")
    infer_config = _load_and_merge_config(args.infer_config, "infer_config")
    measure_config = _load_and_merge_config(args.measure_config, "measure_config")

    logger = configure_logging()

    # -----------------------------------------
    # Instantiate system
    # -----------------------------------------
    system = system_cls(
        train_config=train_config,
        infer_config=infer_config,
        measure_config=measure_config,
    )

    # -----------------------------------------
    # Resolve stages and run
    # -----------------------------------------
    logger.info("System: %s", system_cls.__name__)
    logger.info("Requested stages: %s", args.stages)
    logger.info("Resolved stages: %s", stages_to_run)

    # Guardrail: ensure required configs exist for requested stages
    pretrain_stages = {
        "create_dataset",
        "train_tokenizer",
        "collect_stats",
        "train",
    }
    required_configs = {}
    required_configs.update({stage: train_config for stage in pretrain_stages})
    required_configs.update({"infer": infer_config, "measure": measure_config})
    missing = [
        s
        for s in stages_to_run
        if s in required_configs and required_configs[s] is None
    ]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(
            f"Config not provided for stage(s): {missing_str}. "
            "Use --train_config/--infer_config/--measure_config."
        )
    run_stages(
        system=system,
        stages_to_run=stages_to_run,
        dry_run=args.dry_run,
        log=logger,
        on_stage_start=lambda stage, log: _log_stage_metadata(
            log,
            args=args,
            train_config=train_config,
            infer_config=infer_config,
            measure_config=measure_config,
        ),
    )


def _log_stage_metadata(
    logger,
    args: argparse.Namespace,
    train_config,
    infer_config,
    measure_config,
) -> None:
    log_run_metadata(
        logger,
        argv=sys.argv,
        configs={
            "Training": Path(args.train_config) if args.train_config else None,
            "Inference": Path(args.infer_config) if args.infer_config else None,
            "Metrics": Path(args.measure_config) if args.measure_config else None,
        },
        write_requirements=args.write_requirements,
    )
    log_env_metadata(logger)
    if train_config is not None:
        logger.info(
            "Training config content:\n%s",
            OmegaConf.to_yaml(train_config, resolve=True),
        )
    if infer_config is not None:
        logger.info(
            "Inference config content:\n%s",
            OmegaConf.to_yaml(infer_config, resolve=True),
        )
    if measure_config is not None:
        logger.info(
            "Metrics config content:\n%s",
            OmegaConf.to_yaml(measure_config, resolve=True),
        )


if __name__ == "__main__":
    parser = build_parser(stages=DEFAULT_STAGES)
    args, stages_to_run = parse_cli_and_stage_args(parser, stages=DEFAULT_STAGES)

    # Here you should replace `YourSystemClass` with the actual system class
    # you want to use for your experiment.
    from espnet3.systems.asr.system import ASRSystem 

    main(
        args=args,
        system_cls=ASRSystem,
        stages=DEFAULT_STAGES,
    )

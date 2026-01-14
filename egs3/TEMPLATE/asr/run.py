#!/usr/bin/env python3
"""Generic runner template for System-based experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple, Type

from omegaconf import OmegaConf

from espnet3.utils.config import load_config_with_defaults
from espnet3.utils.logging import configure_logging, log_env_metadata, log_run_metadata
from espnet3.utils.stages import resolve_stages, run_stages

# Default stage list (can be extended/overridden by callers)
DEFAULT_STAGES: List[str] = [
    "create_dataset",
    "train_tokenizer",
    "collect_stats",
    "train",
    "infer",
    "measure",
    "pack_model",
    "upload_model",
    "publish",
]

# Type alias for a System class
SystemCls = Type[Any]
AddArgsFn = Callable[[argparse.ArgumentParser], None]


def build_parser(
    stages: Sequence[str],
    add_arguments: Optional[AddArgsFn] = None,
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
        help="Hydra config for inference/decoding stage.",
    )
    parser.add_argument(
        "--measure_config",
        default=None,
        type=Path,
        help="Hydra config for measure/scoring stage.",
    )
    parser.add_argument(
        "--publish_config",
        default=None,
        type=Path,
        help="Hydra config for pack/upload stages.",
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

    # Stage-specific arguments can be added here if needed
    parser.add_argument(
        "--train.dataset_dir",
        default=None,
        type=Path,
        help="dataset directory for training (used in some stages).",
    )

    # Let caller add custom CLI arguments
    if add_arguments is not None:
        add_arguments(parser)

    return parser


def parse_cli_and_stage_args(
    parser: argparse.ArgumentParser,
    *,
    stages: Sequence[str],
) -> Tuple[argparse.Namespace, List[str]]:
    args = parser.parse_args()
    stages_to_run = resolve_stages(args.stages, stages)

    return args, stages_to_run


def main(
    args,
    system_cls: SystemCls,
    *,
    stages: Sequence[str] = DEFAULT_STAGES,
) -> None:
    stages_to_run = resolve_stages(args.stages, stages)

    # -----------------------------------------
    # Load configs
    # -----------------------------------------
    train_config = (
        None
        if args.train_config is None
        else load_config_with_defaults(args.train_config)
    )
    infer_config = (
        None
        if args.infer_config is None
        else load_config_with_defaults(args.infer_config)
    )
    measure_config = (
        None
        if args.measure_config is None
        else load_config_with_defaults(args.measure_config)
    )
    publish_config = (
        None
        if args.publish_config is None
        else load_config_with_defaults(args.publish_config)
    )
    logger = configure_logging()

    # -----------------------------------------
    # Instantiate system
    # -----------------------------------------
    system = system_cls(
        train_config=train_config,
        infer_config=infer_config,
        measure_config=measure_config,
        publish_config=publish_config,
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
    required_configs.update(
        {
            "pack_model": train_config,
            "upload_model": publish_config,
        }
    )
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
            publish_config=publish_config,
        ),
    )


def _log_stage_metadata(
    logger,
    *,
    args: argparse.Namespace,
    train_config,
    infer_config,
    measure_config,
    publish_config,
) -> None:
    log_run_metadata(
        logger,
        argv=sys.argv,
        configs={
            "train": Path(args.train_config) if args.train_config else None,
            "infer": Path(args.infer_config) if args.infer_config else None,
            "measure": Path(args.measure_config) if args.measure_config else None,
            "publish": Path(args.publish_config) if args.publish_config else None,
        },
        write_requirements=args.write_requirements,
    )
    log_env_metadata(logger)
    if train_config is not None:
        logger.info(
            "Train config content:\n%s", OmegaConf.to_yaml(train_config, resolve=True)
        )
    if infer_config is not None:
        logger.info(
            "Infer config content:\n%s", OmegaConf.to_yaml(infer_config, resolve=True)
        )
    if measure_config is not None:
        logger.info(
            "Measure config content:\n%s",
            OmegaConf.to_yaml(measure_config, resolve=True),
        )
    if publish_config is not None:
        logger.info(
            "Publish config content:\n%s",
            OmegaConf.to_yaml(publish_config, resolve=True),
        )


if __name__ == "__main__":
    parser = build_parser(stages=DEFAULT_STAGES)
    args, stages_to_run = parse_cli_and_stage_args(parser, stages=DEFAULT_STAGES)

    # Here you should replace `YourSystemClass` with the actual system class
    # you want to use for your experiment.
    from espnet3.systems.asr.system import ASRSystem  # Example import

    main(
        args=args,
        system_cls=ASRSystem,
        stages=DEFAULT_STAGES,
    )

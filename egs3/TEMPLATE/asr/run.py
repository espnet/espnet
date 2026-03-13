#!/usr/bin/env python3
"""Generic runner template for System-based experiments."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Sequence

from espnet3.utils.config_utils import load_and_merge_config
from espnet3.utils.logging_utils import configure_logging
from espnet3.utils.stages_utils import (
    parse_cli_and_stage_args,
    resolve_stages,
    run_stages,
)

# Default stage list (can be extended/overridden by callers)
DEFAULT_STAGES: List[str] = [
    "create_dataset",
    "train_tokenizer",
    "collect_stats",
    "train",
    "infer",
    "measure",
]

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
        "--training_config",
        default=None,
        type=Path,
        help=(
            "Hydra config for training (passed to load_config_with_defaults). "
            "Required for create_dataset/train_tokenizer/collect_stats/train stages."
        ),
    )
    parser.add_argument(
        "--inference_config",
        default=None,
        type=Path,
        help="Hydra config for infer stage.",
    )
    parser.add_argument(
        "--metrics_config",
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


def main(
    args,
    system_cls,
    stages: Sequence[str] = DEFAULT_STAGES,
) -> None:
    stages_to_run = resolve_stages(args.stages, stages)

    # -----------------------------------------
    # Load configs
    # -----------------------------------------
    # Keep template_package explicit so the recipe declares which TEMPLATE
    # package provides the default configs, instead of relying on path-based
    # inference from the user-supplied config location.
    training_config = load_and_merge_config(
        args.training_config,
        template_config_path="conf/training.yaml",
        template_package=__package__,
    )
    inference_config = load_and_merge_config(
        args.inference_config,
        template_config_path="conf/inference.yaml",
        template_package=__package__,
    )
    metrics_config = load_and_merge_config(
        args.metrics_config,
        template_config_path="conf/metrics.yaml",
        template_package=__package__,
    )

    logger = configure_logging()

    # -----------------------------------------
    # Instantiate system
    # -----------------------------------------
    system = system_cls(
        training_config=training_config,
        inference_config=inference_config,
        metrics_config=metrics_config,
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
    required_configs.update({stage: training_config for stage in pretrain_stages})
    required_configs.update({"infer": inference_config, "measure": metrics_config})
    missing = [
        s
        for s in stages_to_run
        if s in required_configs and required_configs[s] is None
    ]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(
            f"Config not provided for stage(s): {missing_str}. "
            "Use --training_config/--inference_config/--metrics_config."
        )
    run_stages(
        system=system,
        stages_to_run=stages_to_run,
        args=args,
        log=logger,
    )


if __name__ == "__main__":
    parser = build_parser(stages=DEFAULT_STAGES)
    args, _ = parse_cli_and_stage_args(parser, stages=DEFAULT_STAGES)

    # Here you should replace `YourSystemClass` with the actual system class
    # you want to use for your experiment.
    from espnet3.systems.asr.system import ASRSystem

    main(
        args=args,
        system_cls=ASRSystem,
        stages=DEFAULT_STAGES,
    )

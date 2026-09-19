#!/usr/bin/env python3
"""Generic runner template for enhancement experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence

from espnet3.utils.config_utils import load_and_merge_config
from espnet3.utils.logging_utils import configure_logging
from espnet3.utils.run_utils import (
    apply_training_experiment_context,
    resolve_loaded_configs,
    validate_experiment_context,
)
from espnet3.utils.stages_utils import (
    parse_cli_and_stage_args,
    resolve_stages,
    run_stages,
)

DEFAULT_STAGES: List[str] = [
    "create_dataset",
    "train_tokenizer",
    "collect_stats",
    "train",
    "infer",
    "measure",
]


def build_parser(stages: Sequence[str]) -> argparse.ArgumentParser:
    """Build the command-line parser shared by enhancement recipes."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stages",
        choices=list(stages) + ["all"],
        nargs="+",
        default=["all"],
        help="Which stages to run. Multiple values allowed.",
    )
    parser.add_argument("--training_config", default=None, type=Path)
    parser.add_argument("--inference_config", default=None, type=Path)
    parser.add_argument("--metrics_config", default=None, type=Path)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--write_requirements", action="store_true")
    return parser


def main(args, system_cls, stages: Sequence[str] = DEFAULT_STAGES) -> None:
    """Load enhancement defaults and execute the requested recipe stages."""
    stages_to_run = resolve_stages(args.stages, stages)
    training_config = load_and_merge_config(
        args.training_config,
        config_name="training.yaml",
        default_package=__package__,
        resolve=False,
    )
    inference_config = load_and_merge_config(
        args.inference_config,
        config_name="inference.yaml",
        default_package=__package__,
        resolve=False,
    )
    metrics_config = load_and_merge_config(
        args.metrics_config,
        config_name="metrics.yaml",
        default_package=__package__,
        resolve=False,
    )

    run_logger = configure_logging()
    apply_training_experiment_context(
        training_config=training_config,
        inference_config=inference_config,
        metrics_config=metrics_config,
        log=run_logger,
    )
    validate_experiment_context(
        training_config=training_config,
        inference_config=inference_config,
        metrics_config=metrics_config,
        stages_to_run=stages_to_run,
    )
    resolve_loaded_configs(training_config, inference_config, metrics_config)

    system = system_cls(
        training_config=training_config,
        inference_config=inference_config,
        metrics_config=metrics_config,
    )
    run_logger.info("System: %s", system_cls.__name__)
    run_logger.info("Requested stages: %s", args.stages)
    run_logger.info("Resolved stages: %s", stages_to_run)

    required_configs = {
        "create_dataset": training_config,
        "train_tokenizer": training_config,
        "collect_stats": training_config,
        "train": training_config,
        "infer": inference_config,
        "measure": metrics_config,
    }
    missing = [
        stage
        for stage in stages_to_run
        if stage in required_configs and required_configs[stage] is None
    ]
    if missing:
        raise ValueError(
            f"Config not provided for stage(s): {', '.join(missing)}. "
            "Use --training_config/--inference_config/--metrics_config."
        )

    run_stages(
        system=system,
        stages_to_run=stages_to_run,
        args=args,
        log=run_logger,
    )


if __name__ == "__main__":
    parser = build_parser(stages=DEFAULT_STAGES)
    args, _ = parse_cli_and_stage_args(parser, stages=DEFAULT_STAGES)
    raise RuntimeError("Import this template from a concrete enhancement recipe.")

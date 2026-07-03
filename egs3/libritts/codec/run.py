#!/usr/bin/env python3
"""Runner for the LibriTTS neural codec recipe.

Inlined rather than reusing ``egs3.TEMPLATE.asr.run`` because this recipe's
stage list has no tokenizer stage: codec training only needs the default
ESPnet3 stages (create_dataset, collect_stats, train, infer, measure).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from src.system import CodecSystem

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

DEFAULT_STAGES = [
    "create_dataset",
    "collect_stats",
    "train",
    "infer",
    "measure",
]
DEFAULT_PACKAGE = "egs3.libritts.codec"
DEFAULT_TRAINING_CONFIG = "training_encodec.yaml"
DEFAULT_INFERENCE_CONFIG = "inference.yaml"
DEFAULT_METRICS_CONFIG = "metrics.yaml"


def build_parser(stages: Sequence[str]) -> argparse.ArgumentParser:
    """Build the CLI parser for this recipe."""
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
        help="Hydra config for training-time stages.",
    )
    parser.add_argument(
        "--inference_config",
        default=None,
        type=Path,
        help="Hydra config for the infer stage.",
    )
    parser.add_argument(
        "--metrics_config",
        default=None,
        type=Path,
        help="Hydra config for the measure stage (metrics).",
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


def main(args) -> None:
    stages_to_run = resolve_stages(args.stages, DEFAULT_STAGES)

    training_config = load_and_merge_config(
        args.training_config,
        config_name=DEFAULT_TRAINING_CONFIG,
        default_package=DEFAULT_PACKAGE,
        resolve=False,
    )
    inference_config = load_and_merge_config(
        args.inference_config,
        config_name=DEFAULT_INFERENCE_CONFIG,
        default_package=DEFAULT_PACKAGE,
        resolve=False,
    )
    metrics_config = load_and_merge_config(
        args.metrics_config,
        config_name=DEFAULT_METRICS_CONFIG,
        default_package=DEFAULT_PACKAGE,
        resolve=False,
    )

    logger = configure_logging()
    apply_training_experiment_context(
        training_config=training_config,
        inference_config=inference_config,
        metrics_config=metrics_config,
        log=logger,
    )
    validate_experiment_context(
        training_config=training_config,
        inference_config=inference_config,
        metrics_config=metrics_config,
        stages_to_run=stages_to_run,
    )
    resolve_loaded_configs(training_config, inference_config, metrics_config)

    system = CodecSystem(
        training_config=training_config,
        inference_config=inference_config,
        metrics_config=metrics_config,
    )

    pretrain_stages = {"create_dataset", "collect_stats", "train"}
    required_configs = {stage: training_config for stage in pretrain_stages}
    required_configs["infer"] = inference_config
    required_configs["measure"] = metrics_config
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

    logger.info("System: %s", CodecSystem.__name__)
    logger.info("Requested stages: %s", args.stages)
    logger.info("Resolved stages: %s", stages_to_run)

    run_stages(system=system, stages_to_run=stages_to_run, args=args, log=logger)


if __name__ == "__main__":
    parser = build_parser(stages=DEFAULT_STAGES)
    args, _ = parse_cli_and_stage_args(parser, stages=DEFAULT_STAGES)
    main(args)

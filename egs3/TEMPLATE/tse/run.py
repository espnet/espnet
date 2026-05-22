#!/usr/bin/env python3
"""Generic runner template for TSE System-based experiments."""

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
    "collect_stats",
    "train",
    "infer",
    "measure",
    "pack_model",
    "upload_model",
    "pack_demo",
    "upload_demo",
]


def build_parser(
    stages: Sequence[str],
) -> argparse.ArgumentParser:
    """Build base ArgumentParser for TSE experiments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stages",
        choices=list(stages) + ["all"],
        nargs="+",
        default=list(stages),
        help="Which stages to run. Multiple values allowed.",
    )
    parser.add_argument(
        "--training_config",
        default=None,
        type=Path,
        help=(
            "Config for training. "
            "Required for create_dataset/collect_stats/train stages."
        ),
    )
    parser.add_argument(
        "--inference_config",
        default=None,
        type=Path,
        help="Config for infer stage.",
    )
    parser.add_argument(
        "--metrics_config",
        default=None,
        type=Path,
        help="Config for measure stage.",
    )
    parser.add_argument(
        "--publication_config",
        default=None,
        type=Path,
        help="Config for pack_model/upload_model stages.",
    )
    parser.add_argument(
        "--demo_config",
        default=None,
        type=Path,
        help="Config for pack_demo/upload_demo stages.",
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

    training_config = load_and_merge_config(
        args.training_config,
        config_name="train.yaml",
        default_package=__package__,
        resolve=False,
    )
    inference_config = load_and_merge_config(
        args.inference_config,
        config_name="infer.yaml",
        default_package=__package__,
        resolve=False,
    )
    metrics_config = load_and_merge_config(
        args.metrics_config,
        config_name="measure.yaml",
        default_package=__package__,
        resolve=False,
    )
    publication_config = load_and_merge_config(
        args.publication_config,
        config_name="publish.yaml",
        default_package=__package__,
        resolve=False,
    )
    demo_config = load_and_merge_config(
        args.demo_config,
        config_name="demo.yaml",
        default_package=__package__,
        resolve=False,
    )

    logger = configure_logging()
    apply_training_experiment_context(
        training_config=training_config,
        inference_config=inference_config,
        metrics_config=metrics_config,
        publication_config=publication_config,
        demo_config=demo_config,
        log=logger,
    )
    validate_experiment_context(
        training_config=training_config,
        inference_config=inference_config,
        metrics_config=metrics_config,
        stages_to_run=stages_to_run,
    )
    resolve_loaded_configs(
        training_config,
        inference_config,
        metrics_config,
        publication_config,
        demo_config,
    )

    system = system_cls(
        training_config=training_config,
        inference_config=inference_config,
        metrics_config=metrics_config,
        publication_config=publication_config,
        demo_config=demo_config,
    )

    logger.info("System: %s", system_cls.__name__)
    logger.info("Requested stages: %s", args.stages)
    logger.info("Resolved stages: %s", stages_to_run)

    pretrain_stages = {"create_dataset", "collect_stats", "train"}
    required_configs = {}
    required_configs.update({stage: training_config for stage in pretrain_stages})
    required_configs.update({"infer": inference_config, "measure": metrics_config})
    required_configs.update(
        {
            "pack_model": (training_config, publication_config),
            "upload_model": publication_config,
            "pack_demo": demo_config,
            "upload_demo": demo_config,
        }
    )
    missing = [
        s
        for s in stages_to_run
        if s in required_configs
        and (
            any(cfg is None for cfg in required_configs[s])
            if isinstance(required_configs[s], tuple)
            else required_configs[s] is None
        )
    ]
    if missing:
        raise ValueError(
            f"Config not provided for stage(s): {', '.join(missing)}. "
            "Use --training_config/--inference_config/--metrics_config/"
            "--publication_config/--demo_config."
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

    from espnet3.systems.tse.system import TSESystem

    main(
        args=args,
        system_cls=TSESystem,
        stages=DEFAULT_STAGES,
    )

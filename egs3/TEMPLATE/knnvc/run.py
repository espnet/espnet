#!/usr/bin/env python3
"""Generic runner template for VC System-based experiments."""

from __future__ import annotations

from typing import List, Sequence

from egs3.TEMPLATE.asr.run import build_parser
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

# Default stage list for VC recipes. Two differences from the ASR list:
# - no `train_tokenizer` / `collect_stats`: the vocoder trains on precomputed
#   encoder features written by `prepare_features`;
# - no `pack_demo` / `upload_demo`: a VC demo needs a variable-length set of
#   target-speaker reference utterances as one input, which the shared demo UI
#   asset types (`audio`, `text`) cannot express yet. Add the stages back in a
#   recipe's own `run.py` once a VC UI asset exists.
DEFAULT_STAGES: List[str] = [
    "create_dataset",
    "prepare_features",
    "train",
    "infer",
    "measure",
    "pack_model",
    "upload_model",
]

__all__ = ["DEFAULT_STAGES", "build_parser", "main", "parse_cli_and_stage_args"]


def main(
    args,
    system_cls,
    stages: Sequence[str] = DEFAULT_STAGES,
) -> None:
    """Load configs, instantiate ``system_cls`` and run the requested stages."""
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
    publication_config = load_and_merge_config(
        args.publication_config,
        config_name="publication.yaml",
        default_package=__package__,
        resolve=False,
    )
    if args.demo_config is not None:
        raise ValueError(
            "VC recipes do not support the demo stages yet, so --demo_config "
            "has no effect. See DEFAULT_STAGES in egs3/TEMPLATE/knnvc/run.py."
        )
    demo_config = None
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

    required_configs = {
        "create_dataset": training_config,
        "prepare_features": training_config,
        "train": training_config,
        "infer": inference_config,
        "measure": metrics_config,
        "pack_model": (training_config, publication_config),
        "upload_model": publication_config,
    }
    missing = [
        stage
        for stage in stages_to_run
        if stage in required_configs
        and (
            any(config is None for config in required_configs[stage])
            if isinstance(required_configs[stage], tuple)
            else required_configs[stage] is None
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

    from espnet3.systems.knnvc.system import KNNVCSystem

    main(args=args, system_cls=KNNVCSystem, stages=DEFAULT_STAGES)

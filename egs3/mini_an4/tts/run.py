#!/usr/bin/env python3

from egs3.TEMPLATE.tts.run import build_parser, parse_cli_and_stage_args
from espnet3.systems.tts.system import TTSSystem
from espnet3.utils.config_utils import load_and_merge_config
from espnet3.utils.logging_utils import configure_logging
from espnet3.utils.run_utils import (
    apply_training_experiment_context,
    resolve_loaded_configs,
    validate_experiment_context,
)
from espnet3.utils.stages_utils import resolve_stages, run_stages

DEFAULT_STAGES = [
    "create_dataset",
    "collect_stats",
    "train",
    "infer",
    "pack_model",
    "upload_model",
]

ALL_STAGES = DEFAULT_STAGES
DEFAULT_PACKAGE = "egs3.TEMPLATE.tts"
DEFAULT_TRAINING_CONFIG = "training.yaml"
DEFAULT_INFERENCE_CONFIG = "inference.yaml"


def main(args) -> None:
    stages_to_run = resolve_stages(args.stages, ALL_STAGES)

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
    publish_config = load_and_merge_config(
        args.publish_config,
        config_name="publish.yaml",
        default_package=DEFAULT_PACKAGE,
    )
    demo_config = load_and_merge_config(
        args.demo_config,
        config_name="demo.yaml",
        default_package=DEFAULT_PACKAGE,
    )
    logger = configure_logging()
    apply_training_experiment_context(
        training_config=training_config,
        inference_config=inference_config,
        metrics_config=None,
        log=logger,
    )
    validate_experiment_context(
        training_config=training_config,
        inference_config=inference_config,
        metrics_config=None,
        stages_to_run=stages_to_run,
    )
    resolve_loaded_configs(training_config, inference_config)

    system = TTSSystem(
        training_config=training_config,
        inference_config=inference_config,
        publish_config=publish_config,
        demo_config=demo_config,
    )

    pretrain_stages = {"create_dataset", "collect_stats", "train"}
    required_configs = {stage: training_config for stage in pretrain_stages}
    required_configs.update({"infer": inference_config})
    required_configs.update(
        {
            "pack_model": training_config,
            "upload_model": publish_config,
            "pack_demo": demo_config,
            "upload_demo": demo_config,
        }
    )
    missing = [
        stage
        for stage in stages_to_run
        if stage in required_configs and required_configs[stage] is None
    ]
    if missing:
        raise ValueError(
            f"Config not provided for stage(s): {', '.join(missing)}. "
            "Use --training_config/--inference_config/--publish_config/--demo_config."
        )

    run_stages(system=system, stages_to_run=stages_to_run, args=args, log=logger)


if __name__ == "__main__":
    parser = build_parser(stages=DEFAULT_STAGES)
    args, _ = parse_cli_and_stage_args(parser, stages=DEFAULT_STAGES)
    main(args)

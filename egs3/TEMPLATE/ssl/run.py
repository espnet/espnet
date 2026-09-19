#!/usr/bin/env python3
"""Runner template for BEATs self-supervised pre-training recipes."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Sequence

from omegaconf import DictConfig, OmegaConf

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

# Canonical stage order for one BEATs iteration. Execution always follows this
# order: the tokenizer (iteration > 0) must exist before `infer` writes targets,
# and targets must exist before `collect_stats`/`train` build the dataset.
DEFAULT_STAGES: List[str] = [
    "create_dataset",
    "train_tokenizer",
    "infer",
    "collect_stats",
    "train",
]

# Fields shared by one BEATs iteration. They are set in the encoder training
# config and copied into the other configs, so data locations, iteration,
# teacher, and fbank normalization cannot drift between models.
_SHARED_CONTEXT_KEYS = (
    "recipe_dir",
    "data_dir",
    "dataset_dir",
    "iteration",
    "fbank_mean",
    "fbank_std",
    "waveform_input",
)
TOKENIZER_CONTEXT_KEYS = _SHARED_CONTEXT_KEYS + (
    "stats_dir",
    "ssl_tag",
    "teacher_ckpt_path",
    "num_device",
    "num_nodes",
)
INFERENCE_CONTEXT_KEYS = _SHARED_CONTEXT_KEYS


def build_parser(stages: Sequence[str]) -> argparse.ArgumentParser:
    """Build the BEATs runner argument parser."""
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
        help="Encoder training config. Required by every stage.",
    )
    parser.add_argument(
        "--train_tokenizer_config",
        default=None,
        type=Path,
        help="Tokenizer training config. Required by train_tokenizer and by "
        "infer at iteration > 0.",
    )
    parser.add_argument(
        "--inference_config",
        default=None,
        type=Path,
        help="Tokenization config for the infer stage.",
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


def apply_ssl_training_context(
    training_config: DictConfig | None,
    train_tokenizer_config: DictConfig | None,
    inference_config: DictConfig | None,
    logger: logging.Logger,
) -> None:
    """Copy iteration context from the encoder config into the other configs.

    Values set in ``training_config`` win: :data:`TOKENIZER_CONTEXT_KEYS` are
    copied into ``train_tokenizer_config`` and :data:`INFERENCE_CONTEXT_KEYS`
    into ``inference_config``. Selecting ``--training_config
    conf/training_iter1.yaml`` therefore selects the iteration, teacher, and
    fbank statistics for every stage. Call before ``resolve_loaded_configs``.
    """
    if training_config is None:
        return
    targets = (
        ("train_tokenizer_config", train_tokenizer_config, TOKENIZER_CONTEXT_KEYS),
        ("inference_config", inference_config, INFERENCE_CONTEXT_KEYS),
    )
    for name, config, keys in targets:
        if config is None:
            continue
        for key in keys:
            if training_config.get(key) is None:
                continue
            value = training_config[key]
            if config.get(key) != value:
                logger.info("Setting %s.%s=%r from training_config", name, key, value)
            config[key] = value


def main(args, system_cls, stages: Sequence[str] = DEFAULT_STAGES) -> None:
    """Load configs, validate them for the requested stages, and run stages."""
    stages_to_run = resolve_stages(args.stages, stages)

    training_config = load_and_merge_config(
        args.training_config,
        config_name="training.yaml",
        default_package=__package__,
        resolve=False,
    )
    train_tokenizer_config = load_and_merge_config(
        args.train_tokenizer_config,
        config_name="training_tokenizer.yaml",
        default_package=__package__,
        resolve=False,
    )
    inference_config = load_and_merge_config(
        args.inference_config,
        config_name="inference.yaml",
        default_package=__package__,
        resolve=False,
    )
    logger = configure_logging()

    if training_config is None:
        raise ValueError("--training_config is required for BEATs recipes.")
    missing = []
    if "infer" in stages_to_run and inference_config is None:
        missing.append("infer (--inference_config)")
    iteration = int(training_config.get("iteration", 0) or 0)
    if iteration > 0 and train_tokenizer_config is None:
        if "train_tokenizer" in stages_to_run:
            missing.append("train_tokenizer (--train_tokenizer_config)")
        # infer only needs it to locate the trained tokenizer; an explicit
        # inference_config.model.tokenizer_ckpt_path (external tokenizer) does not.
        external_tokenizer = inference_config is not None and OmegaConf.select(
            inference_config, "model.tokenizer_ckpt_path"
        )
        if "infer" in stages_to_run and not external_tokenizer:
            missing.append("infer (--train_tokenizer_config)")
    if missing:
        raise ValueError("Config not provided for stage(s): " + ", ".join(missing))

    apply_training_experiment_context(
        training_config=training_config,
        inference_config=inference_config,
        metrics_config=None,
        publication_config=None,
        log=logger,
    )
    apply_ssl_training_context(
        training_config, train_tokenizer_config, inference_config, logger
    )
    validate_experiment_context(
        training_config=training_config,
        inference_config=inference_config,
        metrics_config=None,
        stages_to_run=stages_to_run,
    )
    resolve_loaded_configs(training_config, train_tokenizer_config, inference_config)

    system = system_cls(
        training_config=training_config,
        inference_config=inference_config,
        train_tokenizer_config=train_tokenizer_config,
    )
    logger.info("System: %s", system_cls.__name__)
    logger.info("Requested stages: %s", args.stages)
    logger.info("Resolved stages: %s", stages_to_run)
    run_stages(system=system, stages_to_run=stages_to_run, args=args, log=logger)


if __name__ == "__main__":
    from espnet3.systems.ssl.system import BeatsSystem

    parser = build_parser(stages=DEFAULT_STAGES)
    args, _ = parse_cli_and_stage_args(parser, stages=DEFAULT_STAGES)
    main(args=args, system_cls=BeatsSystem, stages=DEFAULT_STAGES)

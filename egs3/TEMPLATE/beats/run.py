#!/usr/bin/env python3
"""Runner template for BEATs pre-training recipes."""

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
from espnet3.utils.stages_utils import run_stages

# Stages of one BEATs iteration that `pretrain` runs in this order. They stay
# available on their own for multi-GPU training, where each training stage must
# be a separate invocation (see BeatsSystem.pretrain).
PRETRAIN_STAGES: List[str] = [
    "create_dataset",
    "train_tokenizer",
    "infer",
    "collect_stats",
    "train",
]

# Canonical execution order of every stage this runner accepts.
ALL_STAGES: List[str] = PRETRAIN_STAGES + [
    "pretrain",
    "measure",
    "pack_model",
    "upload_model",
]

# Stages run by default (and by `--stages all`). There are no demo stages: a
# pre-trained encoder has no user-facing output to demo; downstream recipes
# (e.g. audio classification) own that.
DEFAULT_STAGES: List[str] = ["pretrain", "measure", "pack_model", "upload_model"]

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


def build_parser() -> argparse.ArgumentParser:
    """Build the BEATs runner argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stages",
        choices=ALL_STAGES + ["all"],
        nargs="+",
        default=list(DEFAULT_STAGES),
        help="Stages to run, in canonical order. `all` runs the default stages "
        f"({' '.join(DEFAULT_STAGES)}). `pretrain` runs "
        f"{' '.join(PRETRAIN_STAGES)} and cannot be combined with them.",
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
        "--metrics_config",
        default=None,
        type=Path,
        help="Metrics config for the measure stage.",
    )
    parser.add_argument(
        "--publication_config",
        default=None,
        type=Path,
        help="Publication config for the pack_model/upload_model stages.",
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


def resolve_beats_stages(requested: Sequence[str]) -> List[str]:
    """Resolve ``--stages`` into the stages to run, in canonical order.

    ``all`` expands to :data:`DEFAULT_STAGES`. ``pretrain`` already runs
    :data:`PRETRAIN_STAGES`, so combining it with any of them is rejected
    instead of running those stages twice.

    Examples:
        >>> resolve_beats_stages(["measure", "pretrain"])
        ['pretrain', 'measure']
        >>> resolve_beats_stages(["all"])
        ['pretrain', 'measure', 'pack_model', 'upload_model']

    Raises:
        ValueError: If ``pretrain`` is combined with one of its stages.
    """
    if "all" in requested:
        return list(DEFAULT_STAGES)
    stages = [stage for stage in ALL_STAGES if stage in set(requested)]
    duplicated = [stage for stage in PRETRAIN_STAGES if stage in stages]
    if "pretrain" in stages and duplicated:
        raise ValueError(
            "`pretrain` already runs " + ", ".join(PRETRAIN_STAGES) + "; do not "
            "also request " + ", ".join(duplicated) + "."
        )
    return stages


def apply_beats_training_context(
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
        # Compare unresolved values: the targets hold `???` placeholders and
        # interpolations of them (e.g. `stats_dir: ...${ssl_tag}`).
        current = OmegaConf.to_container(config, resolve=False)
        for key in keys:
            if training_config.get(key) is None:
                continue
            value = training_config[key]
            if current.get(key) != value:
                logger.info("Setting %s.%s=%r from training_config", name, key, value)
            config[key] = value


def _find_missing_configs(
    stages_to_run: Sequence[str],
    iteration: int,
    train_tokenizer_config: DictConfig | None,
    inference_config: DictConfig | None,
    metrics_config: DictConfig | None,
    publication_config: DictConfig | None,
) -> List[str]:
    """Return ``"<stage> (--<flag>)"`` for every stage whose config is missing."""
    runs = set(stages_to_run)
    needs_tokenizer = "pretrain" in runs or "train_tokenizer" in runs
    needs_inference = "pretrain" in runs or "infer" in runs
    missing = []
    if needs_inference and inference_config is None:
        missing.append("infer (--inference_config)")
    if iteration > 0 and train_tokenizer_config is None:
        if needs_tokenizer:
            missing.append("train_tokenizer (--train_tokenizer_config)")
        # infer only needs it to locate the trained tokenizer; an explicit
        # inference_config.model.tokenizer_ckpt_path (external tokenizer) does not.
        external_tokenizer = inference_config is not None and OmegaConf.select(
            inference_config, "model.tokenizer_ckpt_path"
        )
        if needs_inference and not external_tokenizer:
            missing.append("infer (--train_tokenizer_config)")
    if "measure" in runs and metrics_config is None:
        missing.append("measure (--metrics_config)")
    for stage in ("pack_model", "upload_model"):
        if stage in runs and publication_config is None:
            missing.append(f"{stage} (--publication_config)")
    return missing


def main(args, system_cls) -> None:
    """Load configs, validate them for the requested stages, and run stages."""
    stages_to_run = resolve_beats_stages(args.stages)

    def load(path, config_name):
        return load_and_merge_config(
            path,
            config_name=config_name,
            default_package=__package__,
            resolve=False,
        )

    training_config = load(args.training_config, "training.yaml")
    train_tokenizer_config = load(
        args.train_tokenizer_config, "training_tokenizer.yaml"
    )
    inference_config = load(args.inference_config, "inference.yaml")
    metrics_config = load(args.metrics_config, "metrics.yaml")
    publication_config = load(args.publication_config, "publication.yaml")
    logger = configure_logging()

    if training_config is None:
        raise ValueError("--training_config is required for BEATs recipes.")
    missing = _find_missing_configs(
        stages_to_run,
        int(training_config.get("iteration", 0) or 0),
        train_tokenizer_config,
        inference_config,
        metrics_config,
        publication_config,
    )
    if missing:
        raise ValueError("Config not provided for stage(s): " + ", ".join(missing))

    apply_training_experiment_context(
        training_config=training_config,
        inference_config=inference_config,
        metrics_config=metrics_config,
        publication_config=publication_config,
        log=logger,
    )
    apply_beats_training_context(
        training_config, train_tokenizer_config, inference_config, logger
    )
    validate_experiment_context(
        training_config=training_config,
        inference_config=inference_config,
        metrics_config=metrics_config,
        stages_to_run=stages_to_run,
    )
    resolve_loaded_configs(
        training_config,
        train_tokenizer_config,
        inference_config,
        metrics_config,
        publication_config,
    )

    system = system_cls(
        training_config=training_config,
        inference_config=inference_config,
        metrics_config=metrics_config,
        publication_config=publication_config,
        train_tokenizer_config=train_tokenizer_config,
    )
    logger.info("System: %s", system_cls.__name__)
    logger.info("Requested stages: %s", args.stages)
    logger.info("Resolved stages: %s", stages_to_run)
    run_stages(system=system, stages_to_run=stages_to_run, args=args, log=logger)


if __name__ == "__main__":
    from espnet3.systems.beats.system import BeatsSystem

    main(args=build_parser().parse_args(), system_cls=BeatsSystem)

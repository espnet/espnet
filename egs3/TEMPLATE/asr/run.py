# espnet3/utils/run_template.py
#!/usr/bin/env python3
"""Generic runner template for System-based experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type

from espnet3.utils.config import load_config_with_defaults
from espnet3.utils.stages import parse_stage_unknown_args, resolve_stages, run_stages

# Default stage list (can be extended/overridden by callers)
DEFAULT_STAGES: List[str] = [
    "create_dataset",
    "train_tokenizer",
    "collect_stats",
    "train",
    "evaluate",
    "decode",
    "score",
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
        "--stage",
        choices=list(stages) + ["all"],
        nargs="+",
        default=["all"],
        help="Which stages to run. Multiple values allowed.",
    )
    parser.add_argument(
        "--train_config",
        required=True,
        type=Path,
        help="Hydra config name or path for this experiment "
        "(passed to load_config_with_defaults).",
    )
    parser.add_argument(
        "--eval_config",
        default=None,
        type=Path,
        help="Hydra config name or path for evaluation "
        "(passed to load_config_with_defaults).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be executed without actually running stages.",
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
) -> Tuple[argparse.Namespace, Dict[str, Dict[str, Any]], List[str]]:
    args, unknown = parser.parse_known_args()

    stage_configs = parse_stage_unknown_args(
        unknown=unknown,
        valid_stages=stages,
        error=parser.error,
    )

    args.stage_configs = stage_configs
    stages_to_run = resolve_stages(args.stage, stages)

    return args, stage_configs, stages_to_run


def main(
    args,
    system_cls: SystemCls,
    *,
    stages: Sequence[str] = DEFAULT_STAGES,
    stage_configs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    # -----------------------------------------
    # Load configs
    # -----------------------------------------
    train_config = load_config_with_defaults(args.train_config)
    eval_config = (
        None
        if args.eval_config is None
        else load_config_with_defaults(args.eval_config)
    )

    # -----------------------------------------
    # Instantiate system
    # -----------------------------------------
    system = system_cls(
        train_config=train_config,
        eval_config=eval_config,
    )

    # -----------------------------------------
    # Resolve stages and run
    # -----------------------------------------
    stages_to_run = resolve_stages(args.stage, stages)
    run_stages(
        system=system,
        stages_to_run=stages_to_run,
        dry_run=args.dry_run,
        stage_args=stage_configs,
    )


if __name__ == "__main__":
    parser = build_parser(stages=DEFAULT_STAGES)
    args, stage_configs, stages_to_run = parse_cli_and_stage_args(
        parser, stages=DEFAULT_STAGES
    )

    # Here you should replace `YourSystemClass` with the actual system class
    # you want to use for your experiment.
    from espnet3.systems.asr.system import ASRSystem  # Example import

    main(
        args=args,
        system_cls=ASRSystem,
        stages=DEFAULT_STAGES,
        stage_configs=stage_configs,
    )

#!/usr/bin/env python3
"""Convenience script to orchestrate ASR experiment stages via System classes."""

import argparse
from pathlib import Path

from espnet3.systems.asr.system import ASRSystem
from espnet3.utils.config import load_config_with_defaults

STAGES = ["prepare", "train", "evaluate", "decode", "score", "publish"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=STAGES + ["all"],
        nargs="+",
        default=["all"],
        help="Which stages to run. Multiple values allowed.",
    )
    parser.add_argument(
        "--train_config",
        required=True,
        type=Path,
        help="Hydra config name or path for this ASR experiment "
        "(passed to load_config_with_defaults).",
    )
    parser.add_argument(
        "--eval_config",
        default=None,
        type=Path,
        help="Hydra config name or path for this ASR experiment "
        "(passed to load_config_with_defaults).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be executed without actually running stages.",
    )
    args = parser.parse_args()

    # ---------------------------------------------------------
    # Load config (OmegaConf DictConfig) and instantiate system
    # ---------------------------------------------------------
    train_config = load_config_with_defaults(args.train_config)
    eval_config = (
        None
        if args.eval_config is None
        else load_config_with_defaults(args.eval_config)
    )
    system = ASRSystem(train_config=train_config, eval_config=eval_config)

    # ---------------------------------------------------------
    # Resolve stages to run
    # ---------------------------------------------------------
    if "all" in args.stage:
        stages_to_run = STAGES
    else:
        stages_to_run = args.stage

    # ---------------------------------------------------------
    # Run stages in order
    # ---------------------------------------------------------
    for stage in stages_to_run:
        fn = getattr(system, stage, None)
        if fn is None:
            raise AttributeError(f"System has no stage method: {stage}")

        if args.dry_run:
            print(f"[DRY RUN] would run stage: {stage}")
            continue

        print(f"=== Running stage: {stage} ===")
        fn()  # e.g., system.prepare(), system.train(), ...


if __name__ == "__main__":  # pragma: no cover
    main()

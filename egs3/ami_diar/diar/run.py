#!/usr/bin/env python3
"""Run script for AMI diarization recipe.

This script provides a command-line interface for running the diarization pipeline.
"""

import sys
from pathlib import Path

# Add parent directories to path
recipe_dir = Path(__file__).parent
sys.path.insert(0, str(recipe_dir))
sys.path.insert(0, str(recipe_dir.parent.parent.parent))  # espnet root

# Import system
from espnet3.systems.diarization.system import DiarizationSystem

# Import template utilities
sys.path.insert(0, str(recipe_dir.parent.parent / "TEMPLATE" / "asr"))
from run import DEFAULT_STAGES, build_parser, main, parse_cli_and_stage_args

if __name__ == "__main__":
    # Parse arguments
    parser = build_parser()
    args, stage_args = parse_cli_and_stage_args(parser, DEFAULT_STAGES)

    # Create and run system
    system = DiarizationSystem(
        train_config_path=args.train_config,
        infer_config_path=args.infer_config,
        metric_config_path=args.metric_config,
    )

    # Run requested stages
    main(system, args, stage_args)

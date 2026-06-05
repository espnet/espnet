#!/usr/bin/env python3
"""Runner for the Sortformer diarization recipe."""

from typing import List

from egs3.TEMPLATE.asr.run import build_parser, main, parse_cli_and_stage_args
from espnet3.systems.diar.system import DiarizationSystem

DEFAULT_STAGES: List[str] = [
    "data_preparation",
    "collect_stats",
    "train",
    "infer",
    "measure",
    # Full-session (long-form) diarization with the streaming speaker cache.
    "infer_longform",
    "measure_longform",
]

if __name__ == "__main__":
    parser = build_parser(stages=DEFAULT_STAGES)
    args, _ = parse_cli_and_stage_args(parser, stages=DEFAULT_STAGES)
    main(
        args=args,
        system_cls=DiarizationSystem,
        stages=DEFAULT_STAGES,
    )

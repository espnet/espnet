#!/usr/bin/env python3
"""Runner template for speaker verification experiments."""

from __future__ import annotations

from typing import List, Sequence

from egs3.TEMPLATE.asr.run import build_parser
from egs3.TEMPLATE.asr.run import main as run_stages_with_configs
from espnet3.utils.stages_utils import parse_cli_and_stage_args

__all__ = ["DEFAULT_STAGES", "build_parser", "main", "parse_cli_and_stage_args"]

# Speaker verification needs neither a tokenizer nor feature statistics: the
# model is trained on fixed-length raw waveform crops.
DEFAULT_STAGES: List[str] = [
    "create_dataset",
    "train",
    "infer",
    "measure",
    "pack_model",
    "upload_model",
]


def main(
    args,
    system_cls,
    stages: Sequence[str] = DEFAULT_STAGES,
) -> None:
    """Run the requested stages against the speaker default configs."""
    run_stages_with_configs(
        args=args,
        system_cls=system_cls,
        stages=stages,
        default_package=__package__,
    )


if __name__ == "__main__":
    parser = build_parser(stages=DEFAULT_STAGES)
    args, stages_to_run = parse_cli_and_stage_args(parser, stages=DEFAULT_STAGES)

    from espnet3.systems.base.system import BaseSystem

    main(args=args, system_cls=BaseSystem, stages=stages_to_run)

#!/usr/bin/env python3
"""Runner for the 8-speaker streaming Sortformer diarization recipe.

Thin entrypoint that reuses the shared ESPnet3 stage runner
(``build_parser`` / ``main`` from ``egs3.TEMPLATE.asr.run``) with this recipe's
:class:`~espnet3.systems.diar.system.DiarizationSystem` and :data:`DEFAULT_STAGES`.

Run it from the recipe directory, e.g.::

    python run.py --stage data_preparation --stop_stage measure_longform

The default pipeline runs data preparation, training, then long-form inference
and DER scoring on AMI.
"""

from typing import List

from egs3.TEMPLATE.asr.run import build_parser, main, parse_cli_and_stage_args
from espnet3.systems.diar.system import DiarizationSystem

DEFAULT_STAGES: List[str] = [
    # Stage 1: generate FastMSS meetings + build AMI cuts.
    "data_preparation",
    # Stage 2: train the 8-spk streaming Sortformer (cache in the loop).
    "train",
    # Stage 3: long-form (full-session) diarization + collar DER on AMI.
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

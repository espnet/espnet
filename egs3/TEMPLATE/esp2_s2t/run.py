"""Runner template for S2T experiments.

The argument parsing and stage dispatch are the generic ones. They live
under ``egs3/TEMPLATE/asr`` because that is where the first system to need
them put them; nothing in them is ASR-specific.
"""

from __future__ import annotations

from typing import List

from egs3.TEMPLATE.asr.run import build_parser, main, parse_cli_and_stage_args
from espnet3.systems.esp2_s2t.system import S2TSystem

# The generic list minus train_tokenizer: an S2T vocabulary ships with the
# pretrained model, so there is nothing to train. Advertising the stage would
# let --stages reach a method that only raises.
DEFAULT_STAGES: List[str] = [
    "create_dataset",
    "collect_stats",
    "train",
    "infer",
    "measure",
    "pack_model",
    "upload_model",
]

__all__ = [
    "DEFAULT_STAGES",
    "S2TSystem",
    "build_parser",
    "main",
    "parse_cli_and_stage_args",
]

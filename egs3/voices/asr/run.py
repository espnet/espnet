"""Run the VOiCES devkit recipe through the shared ASR stage entrypoint."""

from egs3.TEMPLATE.asr.run import (
    DEFAULT_STAGES,
    build_parser,
    main,
    parse_cli_and_stage_args,
)
from egs3.voices.asr.src.system import VoicesSystem

if __name__ == "__main__":
    parser = build_parser(stages=DEFAULT_STAGES)
    args, _ = parse_cli_and_stage_args(parser, stages=DEFAULT_STAGES)
    main(args=args, system_cls=VoicesSystem, stages=DEFAULT_STAGES)

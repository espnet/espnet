"""Run the full AN4 Sinc-BLSTMP recipe using the shared ASR entrypoint."""

from egs3.an4.asr.src.system import An4System
from egs3.TEMPLATE.asr.run import (
    DEFAULT_STAGES,
    build_parser,
    main,
    parse_cli_and_stage_args,
)

if __name__ == "__main__":
    parser = build_parser(stages=DEFAULT_STAGES)
    args, _ = parse_cli_and_stage_args(parser, stages=DEFAULT_STAGES)
    main(args=args, system_cls=An4System, stages=DEFAULT_STAGES)

"""Entry point for the OWSM v4 recipe; the stages live in the template."""

from egs3.TEMPLATE.owsm.run import (
    DEFAULT_STAGES,
    build_parser,
    main,
    parse_cli_and_stage_args,
)
from espnet3.systems.owsm.system import OWSMSystem

if __name__ == "__main__":
    parser = build_parser(
        stages=DEFAULT_STAGES,
    )
    args, stages_to_run = parse_cli_and_stage_args(parser, stages=DEFAULT_STAGES)

    main(
        args=args,
        system_cls=OWSMSystem,
        stages=stages_to_run,
    )

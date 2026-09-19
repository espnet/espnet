from egs3.TEMPLATE.lid.run import (
    DEFAULT_STAGES,
    build_parser,
    main,
    parse_cli_and_stage_args,
)
from espnet3.systems.lid.system import LIDSystem

if __name__ == "__main__":
    parser = build_parser(stages=DEFAULT_STAGES)
    args, stages_to_run = parse_cli_and_stage_args(parser, stages=DEFAULT_STAGES)

    main(
        args=args,
        system_cls=LIDSystem,
        stages=stages_to_run,
    )

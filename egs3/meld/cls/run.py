from egs3.TEMPLATE.cls.run import (
    DEFAULT_STAGES,
    build_parser,
    main,
    parse_cli_and_stage_args,
)
from espnet3.systems.cls.system import CLSSystem

if __name__ == "__main__":
    parser = build_parser(
        stages=DEFAULT_STAGES,
    )
    args, _ = parse_cli_and_stage_args(parser, stages=DEFAULT_STAGES)

    main(
        args=args,
        system_cls=CLSSystem,
        stages=DEFAULT_STAGES,
    )

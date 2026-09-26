from egs3.TEMPLATE.esp2_cls.run import (
    DEFAULT_STAGES,
    build_parser,
    main,
    parse_cli_and_stage_args,
)
from espnet3.systems.esp2_cls.system import Esp2ClsSystem

if __name__ == "__main__":
    parser = build_parser(
        stages=DEFAULT_STAGES,
    )
    args, _ = parse_cli_and_stage_args(parser, stages=DEFAULT_STAGES)

    main(
        args=args,
        system_cls=Esp2ClsSystem,
        stages=DEFAULT_STAGES,
    )

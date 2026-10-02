from egs3.TEMPLATE.asr.run import (
    DEFAULT_STAGES,
    build_parser,
    main,
    parse_cli_and_stage_args,
)
from espnet3.systems.esp2_slu.system import Esp2SluSystem

if __name__ == "__main__":
    parser = build_parser(stages=DEFAULT_STAGES)
    args, stages_to_run = parse_cli_and_stage_args(parser, stages=DEFAULT_STAGES)
    main(
        args=args,
        system_cls=Esp2SluSystem,
        stages=stages_to_run,
        default_package="egs3.TEMPLATE.esp2_slu",
    )

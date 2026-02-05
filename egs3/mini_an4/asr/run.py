from egs3.TEMPLATE.asr.run import (
    ALL_STAGES,
    build_parser,
    main,
    parse_cli_and_stage_args,
)
from espnet3.systems.asr.system import ASRSystem

if __name__ == "__main__":
    parser = build_parser(
        stages=ALL_STAGES,
        add_arguments=None,
    )
    args, stages_to_run = parse_cli_and_stage_args(parser, stages=ALL_STAGES)

    main(
        args=args,
        system_cls=ASRSystem,
        stages=ALL_STAGES,
    )

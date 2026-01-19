from egs3.TEMPLATE.asr.run import (
    DEFAULT_STAGES,
    ALL_STAGES,
    build_parser,
    main,
    parse_cli_and_stage_args,
)
from espnet3.systems.asr.system import ASRSystem

if __name__ == "__main__":
    parser = build_parser(
        stages=ALL_STAGES,
        default_stages=DEFAULT_STAGES,
        add_arguments=None,
    )
    args, _ = parse_cli_and_stage_args(parser, stages=ALL_STAGES)

    main(
        args=args,
        system_cls=ASRSystem,
        stages=ALL_STAGES,
    )

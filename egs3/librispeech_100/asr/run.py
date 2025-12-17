from egs3.TEMPLATE.asr.run import (
    DEFAULT_STAGES,
    build_parser,
    main,
    parse_cli_and_stage_args,
)
from espnet3.systems.asr.system import ASRSystem  # Example import

if __name__ == "__main__":
    parser = build_parser(
        stages=DEFAULT_STAGES,
        add_arguments=None,  # You can create your parser and put here if needed
    )
    args, stages_to_run = parse_cli_and_stage_args(parser, stages=DEFAULT_STAGES)

    main(
        args=args,
        system_cls=ASRSystem,
        stages=stages_to_run,
    )

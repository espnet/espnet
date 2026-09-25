from egs3.TEMPLATE.esp2_s2t.run import (
    DEFAULT_STAGES,
    S2TSystem,
    build_parser,
    main,
    parse_cli_and_stage_args,
)

if __name__ == "__main__":
    parser = build_parser(
        stages=DEFAULT_STAGES,
    )
    args, stages_to_run = parse_cli_and_stage_args(parser, stages=DEFAULT_STAGES)

    main(
        args=args,
        system_cls=S2TSystem,
        stages=stages_to_run,
    )

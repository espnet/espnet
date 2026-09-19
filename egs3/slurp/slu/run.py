from egs3.slurp.slu.src.system import SLUSystem
from egs3.TEMPLATE.asr.run import (
    DEFAULT_STAGES,
    build_parser,
    main,
    parse_cli_and_stage_args,
)

if __name__ == "__main__":
    parser = build_parser(stages=DEFAULT_STAGES)
    args, stages_to_run = parse_cli_and_stage_args(parser, stages=DEFAULT_STAGES)
    main(args=args, system_cls=SLUSystem, stages=stages_to_run)

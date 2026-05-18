from egs3.TEMPLATE.asr.run import (
    build_parser,
    main,
    parse_cli_and_stage_args,
)
from src.system import TTSSystem

DEFAULT_STAGES = [
    "compute_xvectors",
    "remove_long_short",
    "create_token_list",
    "create_dataset",
    "collect_stats",
    "train",
    "infer",
]

if __name__ == "__main__":
    parser = build_parser(stages=DEFAULT_STAGES)
    args, _ = parse_cli_and_stage_args(parser, stages=DEFAULT_STAGES)
    main(
        args=args,
        system_cls=TTSSystem,
        stages=DEFAULT_STAGES,
    )

"""Run LM statistics and training through the shared ESPnet3 entrypoint."""

from egs3.TEMPLATE.esp2_asr.run import build_parser, main, parse_cli_and_stage_args
from espnet3.systems.esp2_asr.lm_system import LMSystem

DEFAULT_STAGES = ["collect_stats", "train"]

if __name__ == "__main__":
    parser = build_parser(stages=DEFAULT_STAGES)
    args, _ = parse_cli_and_stage_args(parser, stages=DEFAULT_STAGES)
    main(args=args, system_cls=LMSystem, stages=DEFAULT_STAGES)

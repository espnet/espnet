"""ESPnet3 entry point for MuST-C speech translation.

This recipe reproduces the egs2 ST recipe (``egs2/must_c/st1``) with
``STSystem``: separate source and target vocabularies, as ``st.sh`` builds.
"""

from egs3.TEMPLATE.asr.run import (
    DEFAULT_STAGES,
    build_parser,
    main,
    parse_cli_and_stage_args,
)
from espnet3.systems.st.system import STSystem

if __name__ == "__main__":
    parser = build_parser(stages=DEFAULT_STAGES)
    args, _ = parse_cli_and_stage_args(parser, stages=DEFAULT_STAGES)
    main(args=args, system_cls=STSystem, stages=DEFAULT_STAGES)

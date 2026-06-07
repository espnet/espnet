from egs3.TEMPLATE.asr.run import (
    DEFAULT_STAGES,
    build_parser,
    main,
)
from espnet3.systems.asr.system import ASRSystem

if __name__ == "__main__":
    parser = build_parser(stages=DEFAULT_STAGES)
    args = parser.parse_args()

    main(
        args=args,
        system_cls=ASRSystem,
        stages=DEFAULT_STAGES,
    )

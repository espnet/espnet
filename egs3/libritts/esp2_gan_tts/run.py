from egs3.TEMPLATE.esp2_gan_tts.run import (
    DEFAULT_STAGES,
    build_parser,
    main,
    parse_cli_and_stage_args,
)
from espnet3.systems.esp2_gan_tts.system import GANTTSSystem

if __name__ == "__main__":
    parser = build_parser(
        stages=DEFAULT_STAGES,
    )
    args, _ = parse_cli_and_stage_args(parser, stages=DEFAULT_STAGES)

    main(
        args=args,
        system_cls=GANTTSSystem,
        stages=DEFAULT_STAGES,
    )

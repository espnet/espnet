from egs3.TEMPLATE.asr.run import build_parser, main, parse_cli_and_stage_args
from espnet3.systems.base.system import BaseSystem

# DEFAULT_STAGES minus train_tokenizer, which BaseSystem does not implement:
# a Whisper vocabulary is fixed and nothing about it is trained. Advertising it
# would let --stages reach a method that is not there.
STAGES = [
    "create_dataset",
    "collect_stats",
    "train",
    "infer",
    "measure",
    "pack_model",
    "upload_model",
]

if __name__ == "__main__":
    parser = build_parser(
        stages=STAGES,
    )
    args, stages_to_run = parse_cli_and_stage_args(parser, stages=STAGES)

    main(
        args=args,
        system_cls=BaseSystem,
        stages=stages_to_run,
    )

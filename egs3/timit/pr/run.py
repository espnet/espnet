from egs3.TEMPLATE.pr.run import build_parser, main, parse_cli_and_stage_args
from espnet3.systems.pr.system import PRSystem

# This recipe scores a pretrained model, so it has nothing to train and ships no
# conf/training.yaml. Naming the stages here keeps `--stages all` meaningful.
STAGES = ["create_dataset", "infer", "measure"]

if __name__ == "__main__":
    parser = build_parser(stages=STAGES)
    args, stages_to_run = parse_cli_and_stage_args(parser, stages=STAGES)
    main(args=args, system_cls=PRSystem, stages=stages_to_run)

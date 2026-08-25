#!/usr/bin/env python3
from espnet2.tasks.beats import BeatsTask


def get_parser():
    parser = BeatsTask.get_parser()
    return parser


def main(cmd=None):
    """Beats pretraining.

    Example:
        % python beats_train.py asr --print_config --optim adadelta \
                > conf/beats_asr.yaml
        % python beats_train.py --config conf/train_asr.yaml
    """
    BeatsTask.main(cmd=cmd)


if __name__ == "__main__":
    main()

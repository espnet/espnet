#!/usr/bin/env python3
from espnet2.tasks.aqa import AqaTask


def get_parser():
    """Return the audio metric training argument parser."""
    parser = AqaTask.get_parser()
    return parser


def main(cmd=None):
    """Train an audio metric predictor.

    Example:

        % python aqa_train.py --print_config --optim adadelta
        % python aqa_train.py --config conf/train_universa.yaml
    """
    AqaTask.main(cmd=cmd)


if __name__ == "__main__":
    main()

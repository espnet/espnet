#!/usr/bin/env python3
from espnet2.tasks.aqa import AqaTask


def get_parser():
    """Return the AQA training argument parser."""
    parser = AqaTask.get_parser()
    return parser


def main(cmd=None):
    """Train an audio quality assessment model.

    Example:

        % python aqa_train.py --print_config --optim adadelta
        % python aqa_train.py --config conf/train_aqa.yaml
    """
    AqaTask.main(cmd=cmd)


if __name__ == "__main__":
    main()

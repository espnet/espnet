#!/usr/bin/env python3
from espnet2.tasks.uasr import UASRTask


def get_parser():
    parser = UASRTask.get_parser()
    return parser


def main(cmd=None):
    r"""UASR training.

    Example:

        % python uasr_train.py uasr --print_config --optim adadelta \
                > conf/train_uasr.yaml
        % python uasr_train.py --config conf/train_uasr.yaml
    """
    UASRTask.main(cmd=cmd)


if __name__ == "__main__":
    main()

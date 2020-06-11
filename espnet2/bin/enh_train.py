#!/usr/bin/env python3
from espnet2.tasks.frontend import FrontendTask


def get_parser():
    parser = FrontendTask.get_parser()
    return parser


def main(cmd=None):
    r"""Enhancemnet frontend training.

    Example:

        % python enh_train.py asr --print_config --optim adadelta \
                > conf/train_enh.yaml
        % python enh_train.py --config conf/train_enh.yaml
    """
    FrontendTask.main(cmd=cmd)


if __name__ == "__main__":
    main()

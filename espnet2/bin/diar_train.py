#!/usr/bin/env python3
# Licensed under the MIT license.

from espnet2.tasks.diar import DiarizationTask


def get_parser():
    parser = DiarizationTask.get_parser()
    return parser


def main(cmd=None):
    r"""Enhancemnet frontend training.
    Example:
        % python enh_train.py asr --print_config --optim adadelta \
                > conf/train_enh.yaml
        % python enh_train.py --config conf/train_enh.yaml
    """
    DiarizationTask.main(cmd=cmd)


if __name__ == "__main__":
    main()

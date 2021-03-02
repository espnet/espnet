#!/usr/bin/env python3

# Copyright 2021 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

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

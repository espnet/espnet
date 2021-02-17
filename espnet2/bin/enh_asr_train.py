#!/usr/bin/env python3
from espnet2.tasks.enh_asr import EnhASRTask


def get_parser():
    parser = EnhASRTask.get_parser()
    return parser


def main(cmd=None):
    r"""Enh-ASR training.

    Example:
        % python enh_asr_train.py asr --print_config --optim adadelta \
                > conf/train.yaml
        % python enh_asr_train.py --config conf/train.yaml
    """
    EnhASRTask.main(cmd=cmd)


if __name__ == "__main__":
    main()

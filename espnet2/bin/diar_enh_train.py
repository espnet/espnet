#!/usr/bin/env python3
from espnet2.tasks.diar_enh import DiarEnhTask


def get_parser():
    parser = DiarEnhTask.get_parser()
    return parser


def main(cmd=None):
    r"""Diar-Enh training.
    Example:
        % python enh_asr_train.py asr --print_config --optim adadelta \
                > conf/train.yaml
        % python enh_asr_train.py --config conf/train.yaml
    """
    DiarEnhTask.main(cmd=cmd)


if __name__ == "__main__":
    main()

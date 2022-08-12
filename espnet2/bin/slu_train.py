#!/usr/bin/env python3
from espnet2.tasks.slu import SLUTask


def get_parser():
    parser = SLUTask.get_parser()
    return parser


def main(cmd=None):
    r"""SLU training.

    Example:

        % python asr_train.py asr --print_config --optim adadelta \
                > conf/train_asr.yaml
        % python asr_train.py --config conf/train_asr.yaml
    """
    SLUTask.main(cmd=cmd)


if __name__ == "__main__":
    main()

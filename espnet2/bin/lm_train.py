#!/usr/bin/env python3
from espnet2.tasks.lm import LMTask


def get_parser():
    parser = LMTask.add_arguments()
    return parser


def main(cmd=None):
    """

    Example:

        % python lm_train.py asr --print_config --optim adadelta
        % python lm_train.py --config conf/train_asr.yaml
    """
    LMTask.main(cmd=cmd)


if __name__ == "__main__":
    main()

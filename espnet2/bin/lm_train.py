#!/usr/bin/env python3
import sys

from espnet.utils.cli_utils import get_commandline_args
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
    print(get_commandline_args(), file=sys.stderr)
    LMTask.main(cmd=cmd)


if __name__ == '__main__':
    main()

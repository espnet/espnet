#!/usr/bin/env python3
import sys

from espnet.utils.cli_utils import get_commandline_args
from espnet2.tasks.asr import ASRTask


def get_parser():
    parser = ASRTask.add_arguments()
    return parser


def main(cmd=None):
    """

    Example:

        % python asr_train.py asr --print_config --optim adadelta \
                > conf/train_asr.yaml
        % python asr_train.py --config conf/train_asr.yaml
    """
    print(get_commandline_args(), file=sys.stderr)
    ASRTask.main(cmd=cmd)


if __name__ == '__main__':
    main()

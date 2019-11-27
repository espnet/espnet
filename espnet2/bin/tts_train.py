#!/usr/bin/env python3
import sys

from espnet.utils.cli_utils import get_commandline_args
from espnet2.tasks.tts import TTSTask


def get_parser():
    parser = TTSTask.add_arguments()
    return parser


def main(cmd=None):
    """

    Example:

        % python lm_train.py asr --print_config --optim adadelta
        % python lm_train.py --config conf/train_asr.yaml
    """
    print(get_commandline_args(), file=sys.stderr)
    TTSTask.main(cmd=cmd)


if __name__ == '__main__':
    main()

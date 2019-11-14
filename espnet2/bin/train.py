#!/usr/bin/env python3
import sys

import configargparse

from espnet.utils.cli_utils import get_commandline_args
from espnet2.tasks.asr import ASRTask
from espnet2.tasks.lm import LMTask


def not_implemented(args):
    raise NotImplementedError('Not yet')


def get_parser():
    parser = configargparse.ArgumentParser(
        description='Train dnn',
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
    parser.set_defaults(main=None)
    subparsers = parser.add_subparsers(dest='task')

    subparser = subparsers.add_parser(
        'asr', help='ASR model',
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
    ASRTask.add_arguments(subparser)
    subparser.set_defaults(main=ASRTask.main)

    subparser = subparsers.add_parser(
        'lm', help='LM model',
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
    LMTask.add_arguments(subparser)
    subparser.set_defaults(main=LMTask.main)

    subparser = subparsers.add_parser(
        'tts', help='TTS model',
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
    subparser.set_defaults(main=not_implemented)

    return parser


def main(cmd=None):
    """

    Usage:
        python train.py [-h]
        python train.py <sub-command-name> [-h]
        python train.py <sub-command-name> --print_config
        python train.py <sub-command-name> ...

    Examples:
        % python train.py asr --print_config \
            --optim adadelta \
            --encoder_decoder transformer \
                > conf/train_asr.yaml
        % python train.py asr --config conf/train_asr.yaml
    """
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    if args.main is not None:
        # Jump to BaseTask.main()
        args.main(args)
    else:
        parser.print_help(file=sys.stderr)


if __name__ == '__main__':
    main()

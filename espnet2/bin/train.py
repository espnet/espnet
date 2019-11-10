#!/usr/bin/env python3
import sys

import configargparse

from espnet.utils.cli_utils import get_commandline_args
from espnet2.tasks.asr_rnn import ASRRNNTask
from espnet2.tasks.asr_transformer import ASRTransformerTask


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
        'asr_rnn', help='ASR RNN model',
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
    ASRRNNTask.add_arguments(subparser)
    subparser.set_defaults(main=ASRRNNTask.main)

    subparser = subparsers.add_parser(
        'asr_transformer', help='ASR Transformer model',
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
    ASRTransformerTask.add_arguments(subparser)
    subparser.set_defaults(main=ASRTransformerTask.main)

    subparser = subparsers.add_parser(
        'tts_tacotron', help='TTS Tacotron model',
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
    subparser.set_defaults(main=not_implemented)

    return parser


def main(cmd=None):
    """
    Usage:
        python train.py <sub-command-name> [-h ] ...
        python train.py <sub-command-name> --write_config <path>

    Example:
        % python train.py asr_rnn --show_config --optim adadelta > conf/train_asr.yaml
        % python train.py asr_rnn --config conf/train_asr.yaml
    """
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    if args.main is not None:
        args.main(args)
    else:
        parser.print_help(file=sys.stderr)


if __name__ == '__main__':
    main()


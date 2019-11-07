#!/usr/bin/env python3
import sys

import configargparse

from espnet.utils.cli_utils import get_commandline_args
from espnet2.asr.rnn.task import ASRRNNTask
from espnet2.asr.transformer.task import ASRTransformerTask


def not_implemented(args):
    raise NotImplementedError('Not yet')


if __name__ == '__main__':
    print(get_commandline_args(), file=sys.stderr)

    """
    Usage:
        python train.py <sub-command-name> [-h ] ...
        python train.py <sub-command-name> --gen_yaml <path>
    
    Example:
        % python train.py asr_rnn --gen_yaml conf/train_asr.yaml
        % # Modify conf/train_asr.yaml
        % python train.py asr_rnn --config conf/train_asr.yaml
    """

    parser = configargparse.ArgumentParser(description='Train dnn')
    parser.set_defaults(main=None)
    subparsers = parser.add_subparsers(dest='task')

    subparser = subparsers.add_parser('asr_rnn', help='ASR RNN model')
    ASRRNNTask.add_arguments(subparser)
    subparser.set_defaults(main=ASRRNNTask.main)

    subparser = subparsers.add_parser(
        'asr_transformer', help='ASR Transformer model')
    ASRTransformerTask.add_arguments(subparser)
    subparser.set_defaults(main=ASRTransformerTask.main)

    subparser = subparsers.add_parser('asr_rnnt', help='ASR RNNT model')
    subparser.set_defaults(main=not_implemented)

    subparser = subparsers.add_parser('lm_rnn', help='LM RNN model')
    subparser.set_defaults(main=not_implemented)

    subparser = subparsers.add_parser(
        'tts_tacotron', help='TTS Tacotron model')
    subparser.set_defaults(main=not_implemented)

    args = parser.parse_args()
    if args.main is not None:
        args.main(args)
    else:
        parser.print_help()

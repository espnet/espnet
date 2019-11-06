#!/usr/bin/env python3
import configargparse

from espnet2.asr.rnn.task import ASRRNNTask
from espnet2.asr.transformer.task import ASRTransformerTask


def not_implemented():
    raise NotImplementedError('Not yet')


if __name__ == '__main__':
    parser = configargparse.ArgumentParser(description='Train dnn')
    parser.set_defaults(main=None)
    subparsers = parser.add_subparsers(dest='task')

    subparser = subparsers.add_parser('asr_rnn', help='ASR RNN model')
    ASRRNNTask.add_arguments(subparser)
    subparser.set_defaults(main=ASRRNNTask.main)

    subparser = subparsers.add_parser('asr_transformer', help='ASR Transformer model')
    ASRTransformerTask.add_arguments(subparser)
    subparser.set_defaults(main=ASRTransformerTask.main)

    subparser = subparsers.add_parser('asr_rnnt', help='ASR RNNT model')
    subparser.set_defaults(main=not_implemented)

    subparser = subparsers.add_parser('lm_rnn', help='LM RNN model')
    subparser.set_defaults(main=not_implemented)

    subparser = subparsers.add_parser('tts_tacotron', help='TTS Tacotron model')
    subparser.set_defaults(main=not_implemented)

    args = parser.parse_args()
    if args.main is not None:
        args.main()
    else:
        parser.print_help()

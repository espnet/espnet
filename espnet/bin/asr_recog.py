#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


import configargparse
import logging
import os
import random
import sys

import numpy as np


# NOTE: you need this func to generate our sphinx doc
def get_parser():
    parser = configargparse.ArgumentParser(
        description='Transcribe text from speech using a speech recognition model on one CPU or GPU',
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
    # general configuration
    parser.add('--config', is_config_file=True,
               help='Config file path')
    parser.add('--config2', is_config_file=True,
               help='Second config file path that overwrites the settings in `--config`')
    parser.add('--config3', is_config_file=True,
               help='Third config file path that overwrites the settings in `--config` and `--config2`')

    parser.add_argument('--ngpu', type=int, default=0,
                        help='Number of GPUs')
    parser.add_argument('--backend', type=str, default='chainer',
                        choices=['chainer', 'pytorch'],
                        help='Backend library')
    parser.add_argument('--debugmode', type=int, default=1,
                        help='Debugmode')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed')
    parser.add_argument('--verbose', '-V', type=int, default=1,
                        help='Verbose option')
    parser.add_argument('--batchsize', type=int, default=1,
                        help='Batch size for beam search (0: means no batch processing)')
    parser.add_argument('--preprocess-conf', type=str, default=None,
                        help='The configuration file for the pre-processing')
    parser.add_argument('--api', choices=["default", "v2"])
    # task related
    parser.add_argument('--recog-json', type=str,
                        help='Filename of recognition data (json)')
    parser.add_argument('--result-label', type=str, required=True,
                        help='Filename of result label data (json)')
    # model (parameter) related
    parser.add_argument('--model', type=str, required=True,
                        help='Model file parameters to read')
    parser.add_argument('--model-conf', type=str, default=None,
                        help='Model config file')
    parser.add_argument('--num-spkrs', type=int, default=1,
                        choices=[1, 2],
                        help='Number of speakers in the speech')
    # search related
    parser.add_argument('--nbest', type=int, default=1,
                        help='Output N-best hypotheses')
    parser.add_argument('--beam-size', type=int, default=1,
                        help='Beam size')
    parser.add_argument('--penalty', type=float, default=0.0,
                        help='Incertion penalty')
    parser.add_argument('--maxlenratio', type=float, default=0.0,
                        help="""Input length ratio to obtain max output length.
                        If maxlenratio=0.0 (default), it uses a end-detect function
                        to automatically find maximum hypothesis lengths""")
    parser.add_argument('--minlenratio', type=float, default=0.0,
                        help='Input length ratio to obtain min output length')
    parser.add_argument('--ctc-weight', type=float, default=0.0,
                        help='CTC weight in joint decoding')
    # rnnlm related
    parser.add_argument('--rnnlm', type=str, default=None,
                        help='RNNLM model file to read')
    parser.add_argument('--rnnlm-conf', type=str, default=None,
                        help='RNNLM model config file to read')
    parser.add_argument('--word-rnnlm', type=str, default=None,
                        help='Word RNNLM model file to read')
    parser.add_argument('--word-rnnlm-conf', type=str, default=None,
                        help='Word RNNLM model config file to read')
    parser.add_argument('--word-dict', type=str, default=None,
                        help='Word list to read')
    parser.add_argument('--lm-weight', type=float, default=0.1,
                        help='RNNLM weight')
    # streaming related
    parser.add_argument('--streaming-mode', type=str, default=None,
                        choices=['window', 'segment'],
                        help="""Use streaming recognizer for inference.
                        `--batchsize` must be set to 0 to enable this mode""")
    parser.add_argument('--streaming-window', type=int, default=10,
                        help='Window size')
    parser.add_argument('--streaming-min-blank-dur', type=int, default=10,
                        help='Minimum blank duration threshold')
    parser.add_argument('--streaming-onset-margin', type=int, default=1,
                        help='Onset margin')
    parser.add_argument('--streaming-offset-margin', type=int, default=1,
                        help='Offset margin')
    # speech translation related
    parser.add_argument('--tgt-lang', default=False, type=str,
                        help='target language ID (e.g., <en>, <de>, <fr> etc.)')
    return parser


def main(args):
    parser = get_parser()
    args = parser.parse_args(args)

    # logging info
    if args.verbose == 1:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    elif args.verbose == 2:
        logging.basicConfig(level=logging.DEBUG,
                            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    else:
        logging.basicConfig(
            level=logging.WARN, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
        logging.warning("Skip DEBUG/INFO messages")

    # check CUDA_VISIBLE_DEVICES
    if args.ngpu > 0:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cvd is None:
            logging.warning("CUDA_VISIBLE_DEVICES is not set.")
        elif args.ngpu != len(cvd.split(",")):
            logging.error("#gpus is not matched with CUDA_VISIBLE_DEVICES.")
            sys.exit(1)

        # TODO(mn5k): support of multiple GPUs
        if args.ngpu > 1:
            logging.error("The program only supports ngpu=1.")
            sys.exit(1)

    # display PYTHONPATH
    logging.info('python path = ' + os.environ.get('PYTHONPATH', '(None)'))

    # seed setting
    random.seed(args.seed)
    np.random.seed(args.seed)
    logging.info('set random seed = %d' % args.seed)

    # validate rnn options
    if args.rnnlm is not None and args.word_rnnlm is not None:
        logging.error("It seems that both --rnnlm and --word-rnnlm are specified. Please use either option.")
        sys.exit(1)

    # recog
    logging.info('backend = ' + args.backend)
    if args.num_spkrs == 1:
        if args.backend == "chainer":
            from espnet.asr.chainer_backend.asr import recog
            recog(args)
        elif args.backend == "pytorch":
            # Experimental API that supports custom LMs
            if args.api == "v2":
                from espnet.asr.pytorch_backend.recog import recog_v2
                recog_v2(args)
                return
            from espnet.asr.pytorch_backend.asr import recog
            recog(args)
        else:
            raise ValueError("Only chainer and pytorch are supported.")
    elif args.num_spkrs == 2:
        if args.backend == "pytorch":
            from espnet.asr.pytorch_backend.asr_mix import recog
            recog(args)
        else:
            raise ValueError("Only pytorch is supported.")


if __name__ == '__main__':
    main(sys.argv[1:])

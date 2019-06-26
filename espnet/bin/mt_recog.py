#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
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
    # search related
    parser.add_argument('--nbest', type=int, default=1,
                        help='Output N-best hypotheses')
    parser.add_argument('--beam-size', type=int, default=1,
                        help='Beam size')
    parser.add_argument('--penalty', type=float, default=0.1,
                        help='Incertion penalty')
    parser.add_argument('--maxlenratio', type=float, default=3.0,
                        help="""Input length ratio to obtain max output length.
                        If maxlenratio=0.0 (default), it uses a end-detect function
                        to automatically find maximum hypothesis lengths""")
    parser.add_argument('--minlenratio', type=float, default=0.0,
                        help='Input length ratio to obtain min output length')
    parser.add_argument('--ctc-weight', type=float, default=0.0,
                        help='dummy')
    # rnnlm related
    parser.add_argument('--rnnlm', type=str, default=None,
                        help='RNNLM model file to read')
    parser.add_argument('--rnnlm-conf', type=str, default=None,
                        help='RNNLM model config file to read')
    parser.add_argument('--lm-weight', type=float, default=0.0,
                        help='RNNLM weight')
    # multilingual related
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

    # trans
    logging.info('backend = ' + args.backend)
    if args.backend == "chainer":
        raise NotImplementedError("chainer is not supported for MT now.")
        # TODO(hirofumi): support chainer backend
    elif args.backend == "pytorch":
        from espnet.mt.pytorch_backend.mt import trans
        trans(args)
    else:
        raise ValueError("Only chainer and pytorch are supported.")


if __name__ == '__main__':
    main(sys.argv[1:])

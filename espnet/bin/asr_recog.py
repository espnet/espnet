#!/usr/bin/env python
# encoding: utf-8

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging
import os
import sys

from espnet.bin.bin_utils import check_cuda_visible_devices
from espnet.bin.bin_utils import get_recog_argparser
from espnet.bin.bin_utils import set_logging_level
from espnet.bin.bin_utils import set_seed


def main(args):
    parser = get_recog_argparser('asr')
    # general configuration
    parser.add_argument('--debugmode', default=1, type=int,
                        help='Debugmode')
    parser.add_argument('--seed', default=1, type=int,
                        help='Random seed')
    parser.add_argument('--batch-size', default=1, type=int,
                        help='Batch size for beam search (0: means no batch processing)')
    # task related
    parser.add_argument('--recog-json', type=str,
                        help='Filename of recognition data (json)')
    parser.add_argument('--result-label', type=str, required=True,
                        help='Filename of result label data (json)')
    # search related
    parser.add_argument('--nbest', type=int, default=1,
                        help='Output N-best hypotheses')
    parser.add_argument('--beam-size', type=int, default=1,
                        help='Beam size')
    parser.add_argument('--penalty', default=0.0, type=float,
                        help='Incertion penalty')
    parser.add_argument('--ctc-weight', default=0.0, type=float,
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
    parser.add_argument('--lm-weight', default=0.1, type=float,
                        help='RNNLM weight.')
    args = parser.parse_args(args)

    set_logging_level(args.verbose)

    # TODO(mn5k): support of multiple GPUs
    check_cuda_visible_devices(args.ngpu, 1)

    # display PYTHONPATH
    logging.info('python path = ' + os.environ.get('PYTHONPATH', '(None)'))

    set_seed(args.seed)

    # recog
    logging.info('backend = ' + args.backend)
    if args.backend == "chainer":
        from espnet.asr.chainer_backend.asr import recog
        recog(args)
    elif args.backend == "pytorch":
        from espnet.asr.pytorch_backend.asr import recog
        recog(args)
    else:
        raise ValueError("Only chainer and pytorch are supported.")


if __name__ == '__main__':
    main(sys.argv[1:])

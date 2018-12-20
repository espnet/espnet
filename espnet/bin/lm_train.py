#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# This code is ported from the following implementation written in Torch.
# https://github.com/chainer/chainer/blob/master/examples/ptb/train_ptb_custom_loop.py

from __future__ import division
from __future__ import print_function

import logging
import os
import sys

from espnet.bin.bin_utils import check_cuda_visible_devices
from espnet.bin.bin_utils import get_train_argparser
from espnet.bin.bin_utils import set_logging_level
from espnet.bin.bin_utils import set_seed


def main(args):
    parser = get_train_argparser('lm')
    # general configuration
    parser.add_argument('--dict', type=str, required=True,
                        help='Dictionary')
    # task related
    parser.add_argument('--train-label', type=str, required=True,
                        help='Filename of train label data')
    parser.add_argument('--valid-label', type=str, required=True,
                        help='Filename of validation label data')
    parser.add_argument('--test-label', type=str,
                        help='Filename of test label data')
    # LSTMLM training configuration
    parser.add_argument('--opt', default='sgd', type=str,
                        choices=['sgd', 'adam'],
                        help='Optimizer')
    parser.add_argument('--layer', '-l', type=int, default=2,
                        help='Number of hidden layers')
    parser.add_argument('--unit', '-u', type=int, default=650,
                        help='Number of hidden units')
    parser.add_argument('--maxlen', type=int, default=40,
                        help='Batch size is reduced if the input sequence > ML')
    args = parser.parse_args(args)

    set_logging_level(args.verbose)

    check_cuda_visible_devices(args.ngpu)

    # display PYTHONPATH
    logging.info('python path = ' + os.environ.get('PYTHONPATH', '(None)'))

    set_seed(args.seed)

    # load dictionary
    with open(args.dict, 'rb') as f:
        dictionary = f.readlines()
    char_list = [entry.decode('utf-8').split(' ')[0] for entry in dictionary]
    char_list.insert(0, '<blank>')
    char_list.append('<eos>')
    args.char_list_dict = {x: i for i, x in enumerate(char_list)}
    args.n_vocab = len(char_list)

    # train
    logging.info('backend = ' + args.backend)
    if args.backend == "chainer":
        from espnet.lm.chainer_backend.lm import train
        train(args)
    elif args.backend == "pytorch":
        from espnet.lm.pytorch_backend.lm import train
        train(args)
    else:
        raise ValueError("Only chainer and pytorch are supported.")


if __name__ == '__main__':
    main(sys.argv[1:])

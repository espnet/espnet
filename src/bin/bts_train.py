#!/usr/bin/env python

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


import argparse
import logging
import os
import random
import sys

from distutils.util import strtobool

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    # general configuration
    parser.add_argument('--ngpu', default=0, type=int,
                        help='Number of GPUs')
    parser.add_argument('--backend', default='pytorch', type=str,
                        choices=['chainer', 'pytorch'],
                        help='Backend library')
    parser.add_argument('--outdir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--debugmode', default=1, type=int,
                        help='Debugmode')
    parser.add_argument('--seed', default=1, type=int,
                        help='Random seed')
    parser.add_argument('--resume', '-r', default='', nargs='?',
                        help='Resume the training from snapshot')
    parser.add_argument('--minibatches', '-N', type=int, default='-1',
                        help='Process only N minibatches (for debug)')
    parser.add_argument('--verbose', '-V', default=0, type=int,
                        help='Verbose option')
    # task related
    parser.add_argument('--train-feat', type=str, required=True,
                        help='Filename of train feature data (Kaldi scp)')
    parser.add_argument('--valid-feat', type=str, required=True,
                        help='Filename of validation feature data (Kaldi scp)')
    parser.add_argument('--train-label', type=str, required=True,
                        help='Filename of train label data (json)')
    parser.add_argument('--valid-label', type=str, required=True,
                        help='Filename of validation label data (json)')
    # network archtecture
    # encoder
    parser.add_argument('--embed_dim', default=512, type=int,
                        help='Number of dimension of embedding')
    parser.add_argument('--elayers', default=1, type=int,
                        help='Number of encoder layers')
    parser.add_argument('--eunits', '-u', default=512, type=int,
                        help='Number of encoder hidden units')
    parser.add_argument('--econv_layers', default=3, type=int,
                        help='Number of encoder conv layers')
    parser.add_argument('--econv_chans', default=512, type=int,
                        help='Number of encoder conv filter channels')
    parser.add_argument('--econv_filts', default=5, type=int,
                        help='Number of encoder conv filter size')
    # attention
    parser.add_argument('--adim', default=512, type=int,
                        help='Number of attention transformation dimensions')
    parser.add_argument('--aconv-chans', default=32, type=int,
                        help='Number of attention convolution channels \
                        (negative value indicates no location-aware attention)')
    parser.add_argument('--aconv-filts', default=32, type=int,
                        help='Number of attention convolution filters \
                        (negative value indicates no location-aware attention)')
    parser.add_argument('--cumulate_att_w', default=True, type=strtobool,
                        help="Whether or not to cumulate attetion weights")
    # decoder
    parser.add_argument('--dlayers', default=2, type=int,
                        help='Number of decoder layers')
    parser.add_argument('--dunits', default=1024, type=int,
                        help='Number of decoder hidden units')
    parser.add_argument('--prenet_layers', default=2, type=int,
                        help='Number of prenet layers')
    parser.add_argument('--prenet_units', default=256, type=int,
                        help='Number of prenet hidden units')
    parser.add_argument('--postnet_layers', default=5, type=int,
                        help='Number of postnet layers')
    parser.add_argument('--postnet_chans', default=512, type=int,
                        help='Number of postnet conv filter channels')
    parser.add_argument('--postnet_filts', default=5, type=int,
                        help='Number of postnet conv filter size')
    # model (parameter) related
    parser.add_argument('--dropout-rate', default=0.5, type=float,
                        help='Dropout rate')
    # minibatch related
    parser.add_argument('--batch-size', '-b', default=32, type=int,
                        help='Batch size')
    parser.add_argument('--maxlen-in', default=800, type=int, metavar='ML',
                        help='Batch size is reduced if the input sequence length > ML')
    parser.add_argument('--maxlen-out', default=150, type=int, metavar='ML',
                        help='Batch size is reduced if the output sequence length > ML')
    # optimization related
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='Learning rate for optimizer')
    parser.add_argument('--eps', default=1e-6, type=float,
                        help='Epsilon for optimizer')
    parser.add_argument('--weight-decay', default=1e-6, type=float,
                        help='Weight decay coefficient for optimizer')
    parser.add_argument('--epochs', '-e', default=30, type=int,
                        help='Number of maximum epochs')
    parser.add_argument('--grad-clip', default=1, type=float,
                        help='Gradient norm threshold to clip')
    args = parser.parse_args()

    # logging info
    if args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO, format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')
    else:
        logging.basicConfig(
            level=logging.WARN, format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')
        logging.warning('Skip DEBUG/INFO messages')

    # check CUDA_VISIBLE_DEVICES
    if args.ngpu > 0:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cvd is None:
            logging.warn("CUDA_VISIBLE_DEVICES is not set.")
        elif args.ngpu != len(cvd.split(",")):
            logging.error("#gpus is not matched with CUDA_VISIBLE_DEVICES.")
            sys.exit(1)

    # set random seed
    logging.info('random seed = %d' % args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.backend == "pytorch":
        from bts_pytorch import train
        train(args)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()

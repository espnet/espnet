#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import logging
import ConfigParser
import StringIO

# numerical modules
import numpy as np

# from python_speech_features
from python_speech_features import base

# for kaldi io
from kaldi_io_py import write_mat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None,
                        help='config file (Kaldi format)')
    parser.add_argument('--num-mel-bins', type=int, default=23,
                        help='number of triangular mel-frequncy bins')
    parser.add_argument('--num-fft-bins', type=int, default=512,
                        help='number of fft bins')
    parser.add_argument('--sample-frequency', type=int, default=16000,
                        help='waveform data sample frequency')
    parser.add_argument('--low-freq', type=int, default=20,
                        help='low cutoff frequency for mel bins')
    parser.add_argument('--high-freq', type=int, default=None,
                        help='high cutoff frequency for mel bins')
    parser.add_argument('output', metavar='OUT', type=str,
                        help='output filename of Mel-filterbank matrix')
    args = parser.parse_args()

    # config parser without section
    if args.config is not None:
        ini_str = '[root]\n' + open(args.config, 'r').read()
        ini_str = ini_str.replace('--', '').replace('-', '_')  # remove '--' in the kaldi config
        ini_fp = StringIO.StringIO(ini_str)
        config = ConfigParser.RawConfigParser()
        config.readfp(ini_fp)

        # set config file values as defaults
        parser.set_defaults(**dict(config.items('root')))
        args = parser.parse_args()

    if args.high_freq is None:
        args.high_freq = int(args.sample_frequency / 2)

    # logging info
    logging.basicConfig(level=logging.INFO, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    for arg in vars(args):
        logging.info(arg + ": " + str(getattr(args, arg)))

    melmat = base.get_filterbanks(args.num_mel_bins, args.num_fft_bins, args.sample_frequency,
                                  args.low_freq, args.high_freq)

    write_mat(args.output, melmat)

if __name__ == '__main__':
    main()

#!/usr/bin/env python

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
from distutils.util import strtobool
import logging

import kaldiio
import numpy

from espnet.transform.spectrogram import spectrogram
from espnet.utils.cli_utils import FileWriterWrapper
from espnet.utils.cli_utils import get_commandline_args


def get_parser():
    parser = argparse.ArgumentParser(
        description='compute STFT feature from WAV',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n_fft', type=int, default=1024,
                        help='FFT length in point')
    parser.add_argument('--n_shift', type=int, default=512,
                        help='Shift length in point')
    parser.add_argument('--win_length', type=int, default=None, nargs='?',
                        help='Analisys window length in point')
    parser.add_argument('--window', type=str, default='hann',
                        choices=['hann', 'hamming'],
                        help='Type of window')
    parser.add_argument('--write-num-frames', type=str,
                        help='Specify wspecifer for utt2num_frames')
    parser.add_argument('--filetype', type=str, default='mat',
                        choices=['mat', 'hdf5'],
                        help='Specify the file format. '
                             '"mat" is the matrix format in kaldi')
    parser.add_argument('--compress', type=strtobool, default=False,
                        help='Save in compressed format')
    parser.add_argument('--compression-method', type=int, default=2,
                        help='Specify the method(if mat) or gzip-level(if hdf5)')
    parser.add_argument('--verbose', '-V', default=0, type=int,
                        help='Verbose option')
    parser.add_argument('--normalize', choices=[1, 16, 24, 32], type=int,
                        default=None,
                        help='Give the bit depth of the PCM, '
                             'then normalizes data to scale in [-1,1]')
    parser.add_argument('rspecifier', type=str, help='WAV scp file')
    parser.add_argument(
        '--segments', type=str,
        help='segments-file format: each line is either'
             '<segment-id> <recording-id> <start-time> <end-time>'
             'e.g. call-861225-A-0050-0065 call-861225-A 5.0 6.5')
    parser.add_argument('wspecifier', type=str, help='Write specifier')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    if args.verbose > 0:
        logging.basicConfig(level=logging.INFO, format=logfmt)
    else:
        logging.basicConfig(level=logging.WARN, format=logfmt)
    logging.info(get_commandline_args())

    with kaldiio.ReadHelper(args.rspecifier,
                            segments=args.segments) as reader, \
            FileWriterWrapper(args.wspecifier,
                              filetype=args.filetype,
                              write_num_frames=args.write_num_frames,
                              compress=args.compress,
                              compression_method=args.compression_method
                              ) as writer:
        for utt_id, (_, array) in reader:
            array = array.astype(numpy.float32)
            if args.normalize is not None and args.normalize != 1:
                array = array / (1 << (args.normalize - 1))
            spc = spectrogram(
                x=array,
                n_fft=args.n_fft,
                n_shift=args.n_shift,
                win_length=args.win_length,
                window=args.window)
            writer[utt_id] = spc


if __name__ == "__main__":
    main()

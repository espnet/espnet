#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import division

import argparse
import codecs
import logging

import kaldiio
import librosa
import numpy

from espnet.utils.cli_utils import get_commandline_args


def _time_to_str(time_idx):
    time_idx = time_idx * 10 ** 4
    return "%06d" % time_idx


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--fs', type=int,
                        help='Sampling frequency')
    parser.add_argument('--threshold', type=float, default=60,
                        help='Threshold in decibels')
    parser.add_argument('--win_length', type=int, default=1024,
                        help='Analisys window length in point')
    parser.add_argument('--shift_length', type=int, default=256,
                        help='Shift length in point')
    parser.add_argument('--min_silence', type=float, default=0.01,
                        help='minimum silence length')
    parser.add_argument('--verbose', '-V', default=0, type=int,
                        help='Verbose option')
    parser.add_argument('--normalize', choices=[1, 16, 24, 32], type=int,
                        default=None,
                        help='Give the bit depth of the PCM, '
                             'then normalizes data to scale in [-1,1]')
    parser.add_argument('rspecifier', type=str, help='WAV scp file')
    parser.add_argument('wspecifier', type=str, help='Segments file')
    args = parser.parse_args()

    # set logger
    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    if args.verbose > 0:
        logging.basicConfig(level=logging.INFO, format=logfmt)
    else:
        logging.basicConfig(level=logging.WARN, format=logfmt)
    logging.info(get_commandline_args())

    with kaldiio.ReadHelper(args.rspecifier) as reader, \
            codecs.open(args.wspecifier, "w", encoding="utf-8") as f:
        for utt_id, (rate, array) in reader:
            assert rate == args.fs
            array = array.astype(numpy.float32)
            if args.normalize is not None and args.normalize != 1:
                array = array / (1 << (args.normalize - 1))
            array_trim, idx = librosa.effects.trim(
                y=array,
                top_db=args.threshold,
                frame_length=args.win_length,
                hop_length=args.shift_length
            )
            start, end = idx / args.fs

            # added minimum silence part
            start = max(0.0, start - args.min_silence)
            end = min(len(array) / args.fs, end + args.min_silence)

            # write to segments file
            segment = "%s_%s_%s %s %f %f\n" % (
                utt_id, _time_to_str(start), _time_to_str(end),
                utt_id, start, end
            )
            f.write(segment)


if __name__ == "__main__":
    main()

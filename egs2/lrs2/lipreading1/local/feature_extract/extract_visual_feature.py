#!/usr/bin/env python3

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import logging
from distutils.util import strtobool

import kaldiio
import numpy
import resampy
from video_processing import *

from espnet2.utils.types import int_or_none
from espnet.utils.cli_utils import get_commandline_args
from espnet.utils.cli_writers import file_writer_helper


def get_parser():
    parser = argparse.ArgumentParser(
        description="extract visual feature from videos",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--write-num-frames", type=str, help="Specify wspecifer for utt2num_frames"
    )
    parser.add_argument(
        "--filetype",
        type=str,
        default="mat",
        choices=["mat", "hdf5"],
        help="Specify the file format for output. "
        '"mat" is the matrix format in kaldi',
    )
    parser.add_argument(
        "--compress", type=strtobool, default=False, help="Save in compressed format"
    )
    parser.add_argument(
        "--compression-method",
        type=int,
        default=2,
        help="Specify the method(if mat) or " "gzip-level(if hdf5)",
    )
    parser.add_argument("--verbose", "-V", default=0, type=int, help="Verbose option")

    parser.add_argument("rspecifier", type=str, help="WAV scp file")
    parser.add_argument(
        "--segments",
        type=str,
        help="segments-file format: each line is either"
        "<segment-id> <recording-id> <start-time> <end-time>"
        "e.g. call-861225-A-0050-0065 call-861225-A 5.0 6.5",
    )
    parser.add_argument("wspecifier", type=str, help="Write specifier")
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

    with VideoReader(args.rspecifier) as reader, file_writer_helper(
        args.wspecifier,
        filetype=args.filetype,
        write_num_frames=args.write_num_frames,
        compress=args.compress,
        compression_method=args.compression_method,
    ) as writer:
        for utt_id, v_feature in reader:
            writer[utt_id] = v_feature


if __name__ == "__main__":
    main()

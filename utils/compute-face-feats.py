#!/usr/bin/env python3

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
from distutils.util import strtobool
import logging

import kaldiio
import numpy
import resampy

from espnet.transform.video_transform import VideoReader, Lip_Extractor
from espnet.utils.cli_utils import get_commandline_args
from espnet.utils.cli_writers import file_writer_helper
from espnet2.utils.types import int_or_none


def get_parser():
    parser = argparse.ArgumentParser(
        description="compute FACE feature from WAV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--fps", type=int_or_none, help="Video Sampling frequency")
    parser.add_argument("--lip_width", type=int, help="Width of the output lip frame")
    parser.add_argument("--lip_height", type=int, help="Width of the output lip frame")
    parser.add_argument(
            "--shape_predictor_path", 
            type=str, 
            help="the pretrained lip pretector path"
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
    parser.add_argument(
        "--normalize",
        type=bool,
        default=True,
        help="Normalizes image data to scale in [-1,1]",
    )
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
    
    lip_extractor = Lip_Extractor(args.shape_predictor_path)


    reader = VideoReader(
        args.rspecifier,
        args.shape_predictor_path,
        (args.lip_width, args.lip_height)
    )
    with file_writer_helper(
        args.wspecifier,
        filetype=args.filetype,
        write_num_frames=args.write_num_frames,
        compress=args.compress,
        compression_method=args.compression_method,
    ) as writer:
        for utt_id, (rate, lip_frames) in reader:
            if args.fps is not None and rate != args.fps:
                raise Exception("The video sampling rate ({}) is different with the config ({}) !".format(rate, args.fps))


            lip_features = numpy.array(lip_frames)
            t, w, h = lip_features.shape
            lip_features = numpy.reshape(lip_features, (t, w*h))
            if args.normalize:
                lip_features =(lip_features/128) - 1
            writer[utt_id] = lip_features


if __name__ == "__main__":
    main()

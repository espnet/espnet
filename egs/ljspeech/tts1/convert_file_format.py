#!/usr/bin/env python

# (Katsuki Inoue)
# 

import argparse
from distutils.util import strtobool
import os

from espnet.utils.cli_utils import FileReaderWrapper
from espnet.utils.cli_utils import FileWriterWrapper

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--compress', type=strtobool, default=False,
                        help='Save in compressed format')
    parser.add_argument('--write_num_frames', type=str,
                        help='Specify wspecifer for utt2num_frames')
    parser.add_argument('--compression_method', type=int, default=2,
                        help='Specify the method(if mat) or gzip-level(if hdf5)')
    parser.add_argument('--rfiletype', type=str, default='mat',
                        choices=['mat', 'hdf5'],
                        help='Specify the file format for the rspecifier. '
                             '"mat" is the matrix format in kaldi')
    parser.add_argument('--wfiletype', type=str, default='hdf5',
                        choices=['mat', 'hdf5'],
                        help='Specify the file format for the rspecifier. '
                             '"mat" is the matrix format in kaldi')
    parser.add_argument('rspecifier', type=str, help='Input feature')
    parser.add_argument('wspecifier', type=str, help='Write specifier')
    args = parser.parse_args()

#    # check directory
#    if not os.path.exists(args.outdir):
#        os.makedirs(args.outdir)

    with FileWriterWrapper(args.wspecifier,
                           filetype=args.wfiletype,
                           write_num_frames=args.write_num_frames,
                           compress=args.compress,
                           compression_method=args.compression_method
                           ) as writer:
    
        for idx, (utt_id, lmspc) in enumerate(
            FileReaderWrapper(args.rspecifier, args.rfiletype), 1):
    
            print(idx,utt_id)
            writer[utt_id] = lmspc


if __name__ == "__main__":
    main()
#!/usr/bin/env python

import argparse
import logging
import sys

from espnet.utils.cli_utils import get_commandline_args
from espnet.utils.cli_utils import read_rspecifier


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--verbose', '-V', default=0, type=int,
                        help='Verbose option')
    parser.add_argument('--filetype', type=str, default='mat',
                        choices=['mat', 'hdf5'],
                        help='Specify the file format for the rspecifier. '
                             '"mat" is the matrix format in kaldi')
    parser.add_argument('rspecifier', type=str,
                        help='Read specifier for feats. e.g. ark:some.ark')
    parser.add_argument('out', nargs='?', type=argparse.FileType('w'),
                        default=sys.stdout,
                        help='The output filename. '
                             'If omitted, then output to sys.stdout')

    args = parser.parse_args()

    # logging info
    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    if args.verbose > 0:
        logging.basicConfig(level=logging.INFO, format=logfmt)
    else:
        logging.basicConfig(level=logging.WARN, format=logfmt)
    logging.info(get_commandline_args())

    for utt, mat in read_rspecifier(args.rspecifier, args.filetype):
        args.out.write('{} {}\n'.format(utt, ','.join(map(str, mat.shape))))


if __name__ == "__main__":
    main()

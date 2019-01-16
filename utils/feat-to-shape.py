#!/usr/bin/env python
import argparse
from collections import Sequence
import logging
import sys

from espnet.transform.transformation import Transformation
from espnet.utils.cli_utils import FileReaderWrapper
from espnet.utils.cli_utils import get_commandline_args


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--verbose', '-V', default=0, type=int,
                        help='Verbose option')
    parser.add_argument('--filetype', type=str, default='mat',
                        choices=['mat', 'hdf5', 'sound.hdf5', 'sound'],
                        help='Specify the file format for the rspecifier. '
                             '"mat" is the matrix format in kaldi')
    parser.add_argument('--preprocess-conf', type=str, default=None,
                        help='The configuration file for the pre-processing')
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

    if args.preprocess_conf is not None:
        preprocessing = Transformation(args.preprocess_conf)
        logging.info('Apply preprocessing: {}'.format(preprocessing))
    else:
        preprocessing = None

    for utt, mat in FileReaderWrapper(args.rspecifier, args.filetype):
        if isinstance(mat, Sequence):
            # If data is sound file, then got as Tuple[int, ndarray]
            rate, mat = mat
        if preprocessing is not None:
            mat = preprocessing(mat)
        args.out.write('{} {}\n'.format(utt, ','.join(map(str, mat.shape))))


if __name__ == "__main__":
    main()

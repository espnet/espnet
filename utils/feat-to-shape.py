#!/usr/bin/env python

import argparse
import logging
import sys

import h5py
import kaldi_io_py

from espnet.utils.cli_utils import read_hdf5_scp
from espnet.utils.cli_utils import get_commandline_args


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

    if ':' not in args.rspecifier:
        raise RuntimeError('Give "rspecifier" such as "ark:some.ark: {}"'
                           .format(args.rspecifier))
    ftype, filepath = args.rspecifier.split(':', 1)
    if ftype not in ['ark', 'scp']:
        raise RuntimeError('The file type must be one of scp, ark: {}'
                           .format(ftype))
    if args.filetype == 'mat':
        if ftype == 'scp':
            matrices = kaldi_io_py.read_mat_scp(filepath)
        else:
            matrices = kaldi_io_py.read_mat_ark(filepath)

    elif args.filetype == 'hdf5':
        if ftype == 'scp':
            matrices = read_hdf5_scp(filepath)
        else:
            matrices = h5py.File(filepath).items()
    else:
        raise NotImplementedError(
            'Not supporting: --filetype {}'.format(args.filetype))

    for utt, mat in matrices:
        args.out.write('{} {}\n'.format(utt, ','.join(map(str, mat.shape))))


if __name__ == "__main__":
    main()

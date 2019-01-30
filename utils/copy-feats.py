#!/usr/bin/env python
import argparse
from distutils.util import strtobool
import logging

from espnet.transform.transformation import Transformation
from espnet.utils.cli_utils import FileReaderWrapper
from espnet.utils.cli_utils import FileWriterWrapper
from espnet.utils.cli_utils import get_commandline_args
from espnet.utils.cli_utils import is_scipy_wav_style


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--verbose', '-V', default=0, type=int,
                        help='Verbose option')
    parser.add_argument('--in-filetype', type=str, default='mat',
                        choices=['mat', 'hdf5', 'sound.hdf5', 'sound'],
                        help='Specify the file format for the rspecifier. '
                             '"mat" is the matrix format in kaldi')
    parser.add_argument('--out-filetype', type=str, default='mat',
                        choices=['mat', 'hdf5'],
                        help='Specify the file format for the wspecifier. '
                             '"mat" is the matrix format in kaldi')
    parser.add_argument('--write-num-frames', type=str,
                        help='Specify wspecifer for utt2num_frames')
    parser.add_argument('--compress', type=strtobool, default=False,
                        help='Save in compressed format')
    parser.add_argument('--compression-method', type=int, default=2,
                        help='Specify the method(if mat) or gzip-level(if hdf5)')
    parser.add_argument('--preprocess-conf', type=str, default=None,
                        help='The configuration file for the pre-processing')
    parser.add_argument('rspecifier', type=str,
                        help='Read specifier for feats. e.g. ark:some.ark')
    parser.add_argument('wspecifier', type=str,
                        help='Write specifier. e.g. ark:some.ark')
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

    with FileWriterWrapper(
            args.wspecifier,
            filetype=args.out_filetype,
            write_num_frames=args.write_num_frames,
            compress=args.compress,
            compression_method=args.compression_method) as writer:
        for utt, mat in FileReaderWrapper(args.rspecifier, args.in_filetype):
            if is_scipy_wav_style(mat):
                # If data is sound file, then got as Tuple[int, ndarray]
                rate, mat = mat
            if preprocessing is not None:
                mat = preprocessing(mat, uttid_list=utt)
            writer[utt] = mat


if __name__ == "__main__":
    main()

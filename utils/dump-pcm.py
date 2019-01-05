#!/usr/bin/env python
import argparse
from distutils.util import strtobool
import logging

import kaldiio
import numpy

from espnet.utils.cli_utils import FileWriterWrapper
from espnet.utils.cli_utils import get_commandline_args


def wav_generator(rspecifier, segments=None):
    """Generates wav-array from multiple wav-rspecifier

    :param List[str] rspecifier:
    :param str segments:

    """

    readers = [kaldiio.ReadHelper(r, segments=segments) for r in rspecifier]
    for vs in zip(*readers):
        utts = [utt_id for utt_id, _ in vs]
        if not all(utts[i] == utts[0] for i in range(len(vs))):
            raise RuntimeError(
                'The all keys must be common among wav-rspecifiers: {}'
                .format(rspecifier))
        rates = [rate for utt_id, (rate, array) in vs]
        if not all(rates[i] == rates[0] for i in range(len(vs))):
            raise RuntimeError(
                'The all sampling-rage must be common '
                'among wav-rspecifiers: {}'.format(rspecifier))

        arrays = []
        for utt_id, (rate, array) in vs:
            if array.ndim == 1:
                # shape = (Time, 1)
                array = array[:, None]
            arrays.append(array)

        utt_id = utts[0]
        rate = rates[0]

        # [Time, Channel]
        array = numpy.concatenate(arrays, axis=1)
        yield utt_id, (rate, array)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--write-num-frames', type=str,
                        help='Specify wspecifer for utt2num_frames')
    parser.add_argument('--filetype', type=str, default='mat',
                        choices=['mat', 'hdf5', 'flac.hdf5'],
                        help='Specify the file format for output. '
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
    parser.add_argument('rspecifier', type=str, nargs='+', help='WAV scp file')
    parser.add_argument(
        '--segments', type=str,
        help='segments-file format: each line is either'
             '<segment-id> <recording-id> <start-time> <end-time>'
             'e.g. call-861225-A-0050-0065 call-861225-A 5.0 6.5')
    parser.add_argument('wspecifier', type=str, help='Write specifier')
    args = parser.parse_args()

    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    if args.verbose > 0:
        logging.basicConfig(level=logging.INFO, format=logfmt)
    else:
        logging.basicConfig(level=logging.WARN, format=logfmt)
    logging.info(get_commandline_args())

    with FileWriterWrapper(args.wspecifier,
                           filetype=args.filetype,
                           write_num_frames=args.write_num_frames,
                           compress=args.compress,
                           compression_method=args.compression_method
                           ) as writer:
        for utt_id, (rate, array) in wav_generator(args.rspecifier,
                                                   args.segments):
            if args.filetype == 'mat':
                # Kaldi-matrix doesn't support integer
                array = array.astype(numpy.float32)

            if args.normalize is not None and args.normalize != 1:
                array = array.astype(numpy.float32)
                array = array / (1 << (args.normalize - 1))

            # shape = (Time, Channel)
            if args.filetype == 'flac.hdf5':
                writer[utt_id] = (array, rate)
            else:
                writer[utt_id] = array


if __name__ == "__main__":
    main()

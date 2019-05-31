#!/usr/bin/env python
import argparse
from distutils.util import strtobool
import logging

import kaldiio
import numpy

<<<<<<< HEAD
from espnet.transform.transformation import Transformation
=======
>>>>>>> 3c086dddcae725e6068d5dffc26e5962617cf986
from espnet.utils.cli_utils import FileWriterWrapper
from espnet.utils.cli_utils import get_commandline_args


<<<<<<< HEAD
=======
def wav_generator(rspecifier, segments=None):
    """Generates wav-array from multiple wav-rspecifier

    :param List[str] rspecifier:
    :param str segments:

    """

    readers = [kaldiio.ReadHelper(r, segments=segments) for r in rspecifier]
    for vs in zip(*readers):
        for (_, v), r in zip(vs, rspecifier):
            # kaldiio.load_mat can handle both wavfile and kaldi-matrix,
            # and if it is wavfile, returns (rate, ndarray), else ndarray
            if not isinstance(v, tuple):
                raise RuntimeError('"{}" is an invalid wav file.'.format(r))

        utts = [utt_id for utt_id, _ in vs]
        if not all(u == utts[0] for u in utts):
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


>>>>>>> 3c086dddcae725e6068d5dffc26e5962617cf986
def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--write-num-frames', type=str,
                        help='Specify wspecifer for utt2num_frames')
    parser.add_argument('--filetype', type=str, default='mat',
                        choices=['mat', 'hdf5', 'sound.hdf5', 'sound'],
                        help='Specify the file format for output. '
                             '"mat" is the matrix format in kaldi')
<<<<<<< HEAD
    parser.add_argument('--format', type=str, default=None,
                        help='The file format for output pcm. '
                             'This option is only valid '
                             'when "--filetype" is "sound.hdf5" or "sound"')
=======
>>>>>>> 3c086dddcae725e6068d5dffc26e5962617cf986
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
<<<<<<< HEAD
    parser.add_argument('--preprocess-conf', type=str, default=None,
                        help='The configuration file for the pre-processing')
    parser.add_argument('--keep-length', type=strtobool, default=True,
                        help='Truncating or zero padding if the output length '
                             'is changed from the input by preprocessing')
    parser.add_argument('rspecifier', type=str, help='WAV scp file')
=======
    parser.add_argument('rspecifier', type=str, nargs='+', help='WAV scp file')
>>>>>>> 3c086dddcae725e6068d5dffc26e5962617cf986
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

<<<<<<< HEAD
    if args.preprocess_conf is not None:
        preprocessing = Transformation(args.preprocess_conf)
        logging.info('Apply preprocessing: {}'.format(preprocessing))
    else:
        preprocessing = None

=======
>>>>>>> 3c086dddcae725e6068d5dffc26e5962617cf986
    with FileWriterWrapper(args.wspecifier,
                           filetype=args.filetype,
                           write_num_frames=args.write_num_frames,
                           compress=args.compress,
<<<<<<< HEAD
                           compression_method=args.compression_method,
                           pcm_format=args.format
                           ) as writer:
        for utt_id, (rate, array) in kaldiio.ReadHelper(args.rspecifier,
                                                        args.segments):
=======
                           compression_method=args.compression_method
                           ) as writer:
        for utt_id, (rate, array) in wav_generator(args.rspecifier,
                                                   args.segments):
>>>>>>> 3c086dddcae725e6068d5dffc26e5962617cf986
            if args.filetype == 'mat':
                # Kaldi-matrix doesn't support integer
                array = array.astype(numpy.float32)

<<<<<<< HEAD
            if array.ndim == 1:
                # (Time) -> (Time, Channel)
                array = array[:, None]

=======
>>>>>>> 3c086dddcae725e6068d5dffc26e5962617cf986
            if args.normalize is not None and args.normalize != 1:
                array = array.astype(numpy.float32)
                array = array / (1 << (args.normalize - 1))

<<<<<<< HEAD
            if preprocessing is not None:
                orgtype = array.dtype
                out = preprocessing(array, uttid_list=utt_id)
                out = out.astype(orgtype)

                if args.keep_length:
                    if len(out) > len(array):
                        out = numpy.pad(
                            out,
                            [(0, len(out) - len(array))] +
                            [(0, 0) for _ in range(out.ndim - 1)],
                            mode='constant')
                    elif len(out) < len(array):
                        # The length can be changed by stft, for example.
                        out = out[:len(out)]

                array = out

            # shape = (Time, Channel)
            if args.filetype in ['sound.hdf5', 'sound']:
=======
            # shape = (Time, Channel)
            if args.filetype == 'sound.hdf5':
>>>>>>> 3c086dddcae725e6068d5dffc26e5962617cf986
                # Write Tuple[int, numpy.ndarray] (scipy style)
                writer[utt_id] = (rate, array)
            else:
                writer[utt_id] = array


if __name__ == "__main__":
    main()

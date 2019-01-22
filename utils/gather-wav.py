#!/usr/bin/env python
import argparse
import io
import sys

import kaldiio
import numpy
from scipy.io import wavfile

PY2 = sys.version_info[0] == 2


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Gather wav files for each channels into a wav file')
    parser.add_argument(
        'wav', type=str, nargs='+',
        help='Give wav names. You can given them with pipe in kaldi style.'
             'e.g. "some.wav" is equivalent to "cat some.wav |" ')
    parser.add_argument('out', nargs='?', type=argparse.FileType('wb'),
                        default=sys.stdout if PY2 else sys.stdout.buffer,
                        help='The output filename. '
                             'If omitted, then output to sys.stdout')
    args = parser.parse_args()

    rates = []
    signals = []
    for wav_path in args.wav:
        retval = kaldiio.load_mat(wav_path)

        # kaldiio.load_mat can handle both wavfile and kaldi-matrix,
        # and if it is wavfile, returns (rate, ndarray), else ndarray
        if not isinstance(retval, tuple):
            raise RuntimeError('"{}" is an invalid wav file.')
        rate, signal = retval

        if signal.ndim == 1:
            # Change [TIme] -> [Time, Channel]
            signal = signal[:, None]
        rates.append(rate)
        signals.append(signal)

    if not all(r == rates[0] for r in rates):
        raise RuntimeError('Sampling rates are not unified: {}'.format(rates))

    signal = numpy.concatenate(signals, axis=1)
    # scipy.io.wavfile doesn't support stream-output
    f = io.BytesIO()
    wavfile.write(f, rates[0], signal)
    args.out.write(f.getvalue())


if __name__ == '__main__':
    main()

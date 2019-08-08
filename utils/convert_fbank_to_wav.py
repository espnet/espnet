#!/usr/bin/env python

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import logging
import os

import librosa
import numpy as np
from scipy.io.wavfile import write

from espnet.utils.cli_readers import file_reader_helper
from espnet.utils.cli_utils import get_commandline_args


EPS = 1e-10


def logmelspc_to_linearspc(lmspc, fs, n_mels, n_fft, fmin=None, fmax=None):
    """Convert log Mel filterbank to linear spectrogram.

    Args:
        lmspc (ndarray): Log Mel filterbank (T, n_mels).
        fs (int): Sampling frequency.
        n_mels (int): Number of mel basis.
        n_fft (int): Number of FFT points.
        f_min (int, optional): Minimum frequency to analyze.
        f_max (int, optional): Maximum frequency to analyze.

    Returns:
        ndarray: Linear spectrogram (T, n_fft // 2 + 1).

    """
    assert lmspc.shape[1] == n_mels
    fmin = 0 if fmin is None else fmin
    fmax = fs / 2 if fmax is None else fmax
    mspc = np.power(10.0, lmspc)
    mel_basis = librosa.filters.mel(fs, n_fft, n_mels, fmin, fmax)
    inv_mel_basis = np.linalg.pinv(mel_basis)
    spc = np.maximum(EPS, np.dot(inv_mel_basis, mspc.T).T)

    return spc


def griffin_lim(spc, n_fft, n_shift, win_length, window='hann', iters=100):
    """Convert linear spectrogram into waveform using Griffin-Lim.

    Args:
        spc (ndarray): Linear spectrogram (T, n_fft // 2 + 1).
        n_fft (int): Number of FFT points.
        n_shift (int): Shift size in points.
        win_length (int): Window length in points.
        window (str, optional): Window function type.
        iters (int, optionl): Number of iterations of Griffin-Lim Algorithm.

    Returns:
        ndarray: Reconstructed waveform (N,).

    """
    assert spc.shape[1] == n_fft // 2 + 1
    cspc = np.abs(spc).astype(np.complex).T
    angles = np.exp(2j * np.pi * np.random.rand(*cspc.shape))
    y = librosa.istft(cspc * angles, n_shift, win_length, window=window)
    for i in range(iters):
        angles = np.exp(1j * np.angle(librosa.stft(y, n_fft, n_shift, win_length, window=window)))
        y = librosa.istft(cspc * angles, n_shift, win_length, window=window)

    return y


def get_parser():
    parser = argparse.ArgumentParser(
        description='convert FBANK to WAV using Griffin-Lim algorithm',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--fs', type=int, default=22050,
                        help='Sampling frequency')
    parser.add_argument('--fmax', type=int, default=None, nargs='?',
                        help='Maximum frequency')
    parser.add_argument('--fmin', type=int, default=None, nargs='?',
                        help='Minimum frequency')
    parser.add_argument('--n_fft', type=int, default=1024,
                        help='FFT length in point')
    parser.add_argument('--n_shift', type=int, default=512,
                        help='Shift length in point')
    parser.add_argument('--win_length', type=int, default=None, nargs='?',
                        help='Analisys window length in point')
    parser.add_argument('--n_mels', type=int, default=None, nargs='?',
                        help='Number of mel basis')
    parser.add_argument('--window', type=str, default='hann',
                        choices=['hann', 'hamming'],
                        help='Type of window')
    parser.add_argument('--iters', type=int, default=100,
                        help='Number of iterations in Grriffin Lim')
    parser.add_argument('--filetype', type=str, default='mat',
                        choices=['mat', 'hdf5'],
                        help='Specify the file format for the rspecifier. '
                             '"mat" is the matrix format in kaldi')
    parser.add_argument('rspecifier', type=str, help='Input feature')
    parser.add_argument('outdir', type=str,
                        help='Output directory')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    # logging info
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    logging.info(get_commandline_args())

    # check directory
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # extract feature and then write as ark with scp format
    for idx, (utt_id, lmspc) in enumerate(
            file_reader_helper(args.rspecifier, args.filetype), 1):
        if args.n_mels is not None:
            spc = logmelspc_to_linearspc(
                lmspc,
                fs=args.fs,
                n_mels=args.n_mels,
                n_fft=args.n_fft,
                fmin=args.fmin,
                fmax=args.fmax)
        else:
            spc = lmspc
        y = griffin_lim(
            spc,
            n_fft=args.n_fft,
            n_shift=args.n_shift,
            win_length=args.win_length,
            window=args.window,
            iters=args.iters)
        logging.info("(%d) %s" % (idx, utt_id))
        write(args.outdir + "/%s.wav" % utt_id,
              args.fs,
              (y * np.iinfo(np.int16).max).astype(np.int16))


if __name__ == "__main__":
    main()

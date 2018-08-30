#!/usr/bin/env python

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import logging
import os

import librosa
import numpy as np
import soundfile as sf

import kaldi_io_py

EPS = 1e-10


def spectrogram(x, fs, n_fft, n_shift,
                win_length, window='hann'):
    spc = np.abs(librosa.stft(x, n_fft, n_shift, win_length, window=window)).T

    return spc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fs', type=int,
                        help='Sampling frequency')
    parser.add_argument('--n_fft', type=int, default=1024,
                        help='FFT length in point')
    parser.add_argument('--n_shift', type=int, default=512,
                        help='Shift length in point')
    parser.add_argument('--win_length', type=int, default=None, nargs='?',
                        help='Analisys window length in point')
    parser.add_argument('--window', type=str, default='hann',
                        choices=['hann', 'hamming'],
                        help='Type of window')
    parser.add_argument('scp', type=str,
                        help='WAV scp files')
    parser.add_argument('out', type=str,
                        help='Output file id')
    args = parser.parse_args()

    # logging info
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")

    # load scp
    with open(args.scp, 'r') as f:
        scp = [x.replace('\n', '').split() for x in f.readlines()]
    if len(scp[0]) != 2:
        utt_ids = [scp_[0] for scp_ in scp]
        paths = [scp_[-2] for scp_ in scp]
        scp = [[utt_id, path] for utt_id, path in zip(utt_ids, paths)]

    # chech direcitory
    outdir = os.path.dirname(args.out)
    if len(outdir) != 0 and not os.path.exists(outdir):
        os.makedirs(outdir)

    # write to ark and scp file (see https://github.com/vesis84/kaldi-io-for-python)
    arkscp = 'ark:| copy-feats --print-args=false ark:- ark,scp:%s.ark,%s.scp' % (args.out, args.out)

    # extract feature and then write as ark with scp format
    with kaldi_io_py.open_or_fd(arkscp, 'wb') as f:
        for idx, (utt_id, path) in enumerate(scp, 1):
            x, fs = sf.read(path)
            assert fs == args.fs
            spc = spectrogram(
                x=x,
                fs=args.fs,
                n_fft=args.n_fft,
                n_shift=args.n_shift,
                win_length=args.win_length,
                window=args.window)
            logging.info("(%d/%d) %s" % (idx, len(scp), utt_id))
            kaldi_io_py.write_mat(f, spc, utt_id)


if __name__ == "__main__":
    main()

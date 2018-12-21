#!/usr/bin/env python

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import codecs
from distutils.util import strtobool
import logging
import os

import h5py
import librosa
import numpy as np
import soundfile as sf

import kaldi_io_py

from espnet.utils.cli_utils import get_commandline_args

EPS = 1e-10


def spectrogram(x, fs, n_fft, n_shift,
                win_length, window='hann'):
    spc = np.abs(librosa.stft(x, n_fft, n_shift, win_length, window=window)).T

    return spc


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
    parser.add_argument('--write_utt2num_frames', type=strtobool, default=True,
                        help='Whether to write utt2num file')
    parser.add_argument('--filetype', type=str, default='mat',
                        choices=['mat', 'hdf5'],
                        help='Specify the file format. '
                             '"mat" is the matrix format in kaldi')
    parser.add_argument('--compress', type=strtobool, default=False,
                        help='Save in compressed format')
    parser.add_argument('--compression-method', type=int, default=2,
                        help='Specify the method(if mat) or gzip-level(if hdf5)')
    parser.add_argument('--verbose', '-V', default=0, type=int,
                        help='Verbose option')
    parser.add_argument('scp', type=str,
                        help='WAV scp files')
    parser.add_argument('out', type=str,
                        help='Output file id')
    args = parser.parse_args()

    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    if args.verbose > 0:
        logging.basicConfig(level=logging.INFO, format=logfmt)
    else:
        logging.basicConfig(level=logging.WARN, format=logfmt)
    logging.info(get_commandline_args())

    # load scp
    with codecs.open(args.scp, 'r', encoding="utf-8") as f:
        scp = [x.replace('\n', '').split() for x in f.readlines()]
    if len(scp[0]) != 2:
        utt_ids = [scp_[0] for scp_ in scp]
        paths = [scp_[-2] for scp_ in scp]
        scp = [[utt_id, path] for utt_id, path in zip(utt_ids, paths)]

    # check directory
    outdir = os.path.dirname(args.out)
    if len(outdir) != 0 and not os.path.exists(outdir):
        os.makedirs(outdir)

    if args.filetype == 'mat':
        # write to ark and scp file (see https://github.com/vesis84/kaldi-io-for-python)
        if args.write_utt2num_frames:
            job_id = "." + args.out.split(".")[-1] if args.out.split(".")[-1].isdigit() else ""
            arkscp = ('ark:| copy-feats --print-args=false --write-num-frames=ark,t:%s '
                      'ark:- ark,scp:%s.ark,%s.scp') % (
                          os.path.dirname(args.out) + "/utt2num_frames" + job_id, args.out, args.out)
        else:
            arkscp = 'ark:| copy-feats --print-args=false ark:- ark,scp:%s.ark,%s.scp' % (args.out, args.out)
        if args.compress:
            arkscp = arkscp.replace(
                'copy-feats',
                'copy-feats --compress={} --compression-method={}'
                .format(args.compress, args.compression_method))

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

    elif args.filetype == 'hdf5':
        if args.write_utt2num_frames:
            job_id = "." + args.out.split(".")[-1] \
                if args.out.split(".")[-1].isdigit() else ""
            utt2num_frames = open(
                os.path.dirname(args.out) + "/utt2num_frames" + job_id, 'w')
        else:
            utt2num_frames = None

        with h5py.File(args.out + '.h5', 'w') as f, \
                open(args.out + '.scp', 'w') as fscp:
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

                if args.compress:
                    kwargs = dict(compression='gzip',
                                  compression_opts=args.compression_method)
                else:
                    kwargs = {}
                f.create_dataset(utt_id, data=spc, **kwargs)
                fscp.write('{} {}.h5:{}\n'.format(utt_id, args.out, utt_id))
                if utt2num_frames is not None:
                    utt2num_frames.write('{} {}\n'.format(utt_id, len(spc)))

        if utt2num_frames is not None:
            utt2num_frames.close()
    else:
        raise NotImplementedError(
            'Not supporting: --filetype {}'.format(args.filetype))


if __name__ == "__main__":
    main()

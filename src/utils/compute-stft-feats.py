#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import sys
import argparse
import logging
import ConfigParser
import StringIO

# numerical modules
import numpy as np
import scipy.io.wavfile as wav

# from python_speech_features
from python_speech_features import sigproc

# for kaldi io
import kaldi_io


def cspec(signal, samplerate=16000, winlen=0.025, winstep=0.01,
          nfft=512, preemph=0.97,
          winfunc=lambda x: np.ones((x,))):
    """Compute STFT coeeficients from an audio signal.
    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the samplerate of the signal we are working with.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param nfft: the FFT size. Default is 512.
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied.
        You can use numpy window functions here e.g. winfunc=np.hamming
    :returns: 2 values. The first is a numpy array of size (NUMFRAMES by nfilt) containing features.
        Each row holds 1 feature vector. The second return value is the energy in each frame (total energy, unwindowed)
    """
    signal = sigproc.preemphasis(signal, preemph)
    frames = sigproc.framesig(signal, winlen*samplerate, winstep*samplerate, winfunc)
    if np.shape(frames)[1] > nfft:
        logging.warn('frame length (%d) is greater than FFT size (%d), frame will be truncated. '
                     + 'Increase NFFT to avoid.', np.shape(frames)[1], nfft)

    return np.fft.rfft(frames, nfft)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None,
                        help='config file (Kaldi format)')
    parser.add_argument('--frame-length', type=float, default=25,
                        help='Frame length in milliseconds (float, default = 25)')
    parser.add_argument('--frame-shift', type=float, default=10,
                        help='Frame shift in milliseconds (float, default = 10)')
    parser.add_argument('--window-type', type=str, default='hamming',
                        help='Type of window ("hamming"|"hanning") (string, default = "hamming")')
    parser.add_argument('--complex-format', type=str, default='real-imaginary',
                        help='Format of complex numbers ("real-imaginary"|"magnitude-phase") '
                             + '(string, default = "real-imaginary")')
    parser.add_argument('wav_scp', metavar='IN', type=str,
                        help='WAV scp files (do not accept command line)')
    parser.add_argument('feats_wspecifier', metavar='OUT', type=str,
                        help='<feats-wspecifier>')
    args = parser.parse_args()

    # config parser without section
    if args.config is not None:
        ini_str = '[root]\n' + open(args.config, 'r').read()
        ini_str = ini_str.replace('--', '').replace('-', '_')  # remove '--' in the kaldi config
        ini_fp = StringIO.StringIO(ini_str)
        config = ConfigParser.RawConfigParser()
        config.readfp(ini_fp)

        # set config file values as defaults
        parser.set_defaults(**dict(config.items('root')))
        args = parser.parse_args()

    # logging info
    logging.basicConfig(level=logging.INFO, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    for arg in vars(args):
        logging.info(arg + ": " + str(getattr(args, arg)))

    with open(args.wav_scp, 'r') as f:
        scp = [x.split() for x in f.readlines()]  # list of [utt_id, wav_name]

    writer = kaldi_io.BaseFloatMatrixWriter(args.feats_wspecifier)

    for x in scp:
        if len(x) != 2:
            sys.exit("wav.scp must be (utt_id, WAV)")
        (rate, sig) = wav.read(x[1])
        feat = cspec(sig, samplerate=rate)
        if args.complex_format is 'real-imaginary':
            feat = np.hstack((feat.real, feat.imag))
        elif args.complex_format is 'magnitude-phase':
            feat = np.hstack((feat.absolute, feat.angles))
        else:
            sys.exit("do not support a complex number format of " + args.complex_format)
        writer.write(x[0], feat)


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""This code is based on https://github.com/kan-bayashi/PytorchWaveNetVocoder."""

# Copyright 2019 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import logging
import os
import sys
import time

import h5py
import numpy as np
import torch

from scipy.io.wavfile import write
from sklearn.preprocessing import StandardScaler

from espnet.nets.pytorch_backend.wavenet import decode_mu_law
from espnet.nets.pytorch_backend.wavenet import encode_mu_law
from espnet.nets.pytorch_backend.wavenet import WaveNet
from espnet.utils.cli_readers import file_reader_helper
from espnet.utils.cli_utils import get_commandline_args

try:
    from sprocket.speech import Synthesizer
except ImportError:
    logging.error("sprocket-vc is not installed. please install via `. ./path.sh && pip install sprocket-vc`.")
    sys.exit(1)


class NoiseShaper(object):
    """Noise shaper.

    This module apply a noise shaping filter based on `An investigation of noise shaping with perceptual weighting
    for WaveNet-based speech generation`_.

    Args:
        mlsa_coef (ndaaray): Coefficient vector of MLSA filter (D,). Basically, average of mel-cepstrum.
        fs (int, optional): Sampling frequency.
        n_fft (int, optional): Number of points in FFT.
        n_shift (int, optional): Shift length in points.
        mag (float, optional): Magnification of noise shaping.
        alpha (float, optional): Alpha of mel cepstrum.

    .. _`An investigation of noise shaping with perceptual weighting for WaveNet-based speech generation`:
        https://ieeexplore.ieee.org/abstract/document/8461332

    """

    def __init__(self, mlsa_coef, fs=22050, n_fft=1024, n_shift=256, mag=0.5, alpha=None):
        self.mlsa_coef = mlsa_coef * mag
        self.mlsa_coef[0] = 0.0
        self.fs = fs
        self.shiftms = n_shift / fs * 1000
        self.n_fft = n_fft
        if alpha is None:
            if self.fs == 16000:
                self.alpha = 0.42
            else:
                self.alpha = 0.455
        self.synthesizer = Synthesizer(
            fs=self.fs,
            shiftms=self.shiftms,
            fftl=self.n_fft,
        )

    def __call__(self, y):
        """Apply noise shaping filter.

        Args:
            y (ndarray): Wavnform signal normalized from -1 to 1 (N,).

        Returns:
            y (ndarray): Noise shaped wavnform signal normalized from -1 to 1 (N,).

        """
        # check shape and type
        assert len(y.shape) == 1
        y = np.float64(y)

        # get frame number and then replicate mlsa coef
        num_frames = int(1000 * len(y) / self.fs / self.shiftms) + 1
        mlsa_coef = np.float64(np.tile(self.mlsa_coef, [num_frames, 1]))

        # apply mlsa filter
        y = self.synthesizer.synthesis_diff(y, mlsa_coef, alpha=self.alpha)

        return y


def get_parser():
    parser = argparse.ArgumentParser(
        description='generate wav from FBANK to WAV using wavenet vocoder',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--fs', type=int, default=22050,
                        help='Sampling frequency')
    parser.add_argument('--n_fft', type=int, default=1024,
                        help='FFT length in point')
    parser.add_argument('--n_shift', type=int, default=256,
                        help='Shift length in point')
    parser.add_argument('--model', type=str, default=None,
                        help='WaveNet model')
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

    # load model config
    model_dir = os.path.dirname(args.model)
    config = torch.load(model_dir + "/model.conf")

    # load statistics
    scaler = StandardScaler()
    with h5py.File(model_dir + "/stats.h5") as f:
        scaler.mean_ = f["/melspc/mean"][()]
        scaler.scale_ = f["/melspc/scale"][()]

    # load MLSA coef
    with h5py.File(model_dir + "/stats.h5") as f:
        mlsa_coef = f["mcep/mean"][()]

    # define noise shaper
    noise_shaper = NoiseShaper(
        mlsa_coef=mlsa_coef,
        fs=args.fs,
        n_fft=args.n_fft,
        n_shift=args.n_shift,
    )

    # define model and laod parameters
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = WaveNet(
        n_quantize=config.n_quantize,
        n_aux=config.n_aux,
        n_resch=config.n_resch,
        n_skipch=config.n_skipch,
        dilation_depth=config.dilation_depth,
        dilation_repeat=config.dilation_repeat,
        kernel_size=config.kernel_size,
        upsampling_factor=config.upsampling_factor
    )
    model.load_state_dict(torch.load(
        args.model,
        map_location=lambda storage,
        loc: storage)["model"])
    model.eval()
    model.to(device)

    for idx, (utt_id, lmspc) in enumerate(
            file_reader_helper(args.rspecifier, args.filetype), 1):
        logging.info("(%d) %s" % (idx, utt_id))

        # perform preprocesing
        x = encode_mu_law(np.zeros((1)), mu=config.n_quantize)  # quatize initial seed waveform
        h = scaler.transform(lmspc)  # normalize features

        # convert to tensor
        x = torch.tensor(x, dtype=torch.long, device=device)  # (1, )
        h = torch.tensor(h, dtype=torch.float, device=device)  # (T, n_aux)

        # get length of waveform
        n_samples = (h.shape[0] - 1) * args.n_shift + args.n_fft

        # generate
        start_time = time.time()
        with torch.no_grad():
            y = model.generate(x, h, n_samples, interval=100)
        logging.info("generation speed = %s (sec / sample)" % ((time.time() - start_time) / (len(y) - 1)))
        y = decode_mu_law(y, mu=config.n_quantize)

        # applay noise shaping
        y = noise_shaper(y)

        # save as .wav file
        write(args.outdir + "/%s.wav" % utt_id,
              args.fs,
              (y * np.iinfo(np.int16).max).astype(np.int16))


if __name__ == "__main__":
    main()

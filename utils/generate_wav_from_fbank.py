#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""This code is based on https://github.com/kan-bayashi/PytorchWaveNetVocoder."""

# Copyright 2019 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import logging
import os
import time

import h5py
import numpy as np
import pysptk
import torch

from scipy.io.wavfile import write
from sklearn.preprocessing import StandardScaler

from espnet.nets.pytorch_backend.wavenet import decode_mu_law
from espnet.nets.pytorch_backend.wavenet import encode_mu_law
from espnet.nets.pytorch_backend.wavenet import WaveNet
from espnet.utils.cli_readers import file_reader_helper
from espnet.utils.cli_utils import get_commandline_args


class TimeInvariantMLSAFilter(object):
    """Time invariant MLSA filter.

    This module is used to perform noise shaping described in `An investigation of noise shaping with perceptual
    weighting for WaveNet-based speech generation`_.

    Args:
        coef (ndaaray): MLSA filter coefficient (D,).
        alpha (float): All pass constant value.
        n_shift (int): Shift length in points.

    .. _`An investigation of noise shaping with perceptual weighting for WaveNet-based speech generation`:
        https://ieeexplore.ieee.org/abstract/document/8461332

    """

    def __init__(self, coef, alpha, n_shift):
        self.n_shift = n_shift
        self.mlsa_filter = pysptk.synthesis.Synthesizer(
            pysptk.synthesis.MLSADF(
                order=coef.shape[0] - 1,
                alpha=alpha),
            hopsize=n_shift
        )

    def __call__(self, y):
        """Apply time invariant MLSA filter.

        Args:
            y (ndarray): Wavnform signal normalized from -1 to 1 (N,).

        Returns:
            y (ndarray): Filtered waveform signal normalized from -1 to 1 (N,).

        """
        # check shape and type
        assert len(y.shape) == 1
        y = np.float64(y)

        # get frame number and then replicate mlsa coef
        num_frames = int(len(y) / self.n_shift) + 1
        coef = np.tile(self.coef, [num_frames, 1])

        return self.mlsa_filter.synthesis(y, coef)


def get_parser():
    parser = argparse.ArgumentParser(
        description='generate wav from FBANK using wavenet vocoder',
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
    parser.add_argument('rspecifier', type=str, help='Input feature e.g. scp:feat.scp')
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
    train_args = torch.load(os.path.join(model_dir, "model.conf"))

    # load statistics
    scaler = StandardScaler()
    with h5py.File(os.path.join(model_dir, "stats.h5")) as f:
        scaler.mean_ = f["/melspc/mean"][()]
        scaler.scale_ = f["/melspc/scale"][()]
        # TODO(kan-bayashi): include following info as default
        coef = f["/mlsa/coef"][()]
        alpha = f["/mlsa/alpha"][()]

    # define MLSA filter for noise shaping
    mlsa_filter = TimeInvariantMLSAFilter(
        coef=coef,
        alpha=alpha,
        n_shift=args.n_shift,
    )

    # define model and laod parameters
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = WaveNet(
        n_quantize=train_args.n_quantize,
        n_aux=train_args.n_aux,
        n_resch=train_args.n_resch,
        n_skipch=train_args.n_skipch,
        dilation_depth=train_args.dilation_depth,
        dilation_repeat=train_args.dilation_repeat,
        kernel_size=train_args.kernel_size,
        upsampling_factor=train_args.upsampling_factor
    )
    model.load_state_dict(
        torch.load(args.model, map_location="cpu")["model"])
    model.eval()
    model.to(device)

    for idx, (utt_id, lmspc) in enumerate(
            file_reader_helper(args.rspecifier, args.filetype), 1):
        logging.info("(%d) %s" % (idx, utt_id))

        # perform preprocesing
        x = encode_mu_law(np.zeros((1)), mu=train_args.n_quantize)  # quatize initial seed waveform
        h = scaler.transform(lmspc)  # normalize features

        # convert to tensor
        x = torch.tensor(x, dtype=torch.long, device=device)  # (1,)
        h = torch.tensor(h, dtype=torch.float, device=device)  # (T, n_aux)

        # get length of waveform
        n_samples = (h.shape[0] - 1) * args.n_shift + args.n_fft

        # generate
        start_time = time.time()
        with torch.no_grad():
            y = model.generate(x, h, n_samples, interval=100)
        logging.info("generation speed = %s (sec / sample)" % ((time.time() - start_time) / (len(y) - 1)))
        y = decode_mu_law(y, mu=train_args.n_quantize)

        # apply mlsa filter for noise shaping
        y = mlsa_filter(y)

        # save as .wav file
        write(os.path.join(args.outdir, "%s.wav" % utt_id),
              args.fs,
              (y * np.iinfo(np.int16).max).astype(np.int16))


if __name__ == "__main__":
    main()

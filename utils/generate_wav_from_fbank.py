#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import logging
import os
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


def get_parser():
    parser = argparse.ArgumentParser(
        description='generate wav from FBANK to WAV using wavenet vocoder',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--fs', type=int, default=22050,
                        help='Sampling frequency')
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

    # load model
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

    # extract feature and then write as ark with scp format
    for idx, (utt_id, lmspc) in enumerate(
            file_reader_helper(args.rspecifier, args.filetype), 1):
        logging.info("(%d) %s" % (idx, utt_id))

        # perform preprocesing
        x = encode_mu_law(np.zeros((1)), mu=config.n_quantize)  # quatize initial seed waveform
        h = scaler.transform(lmspc)  # normalize features

        # convert to tensor
        x = torch.tensor(x, dtype=torch.long, device=device)  # (1, )
        h = torch.tensor(h, dtype=torch.float, device=device)  # (T, n_aux)

        # generate
        start_time = time.time()
        with torch.no_grad():
            y = model.generate(x, h, interval=100)
        logging.info("generation speed = %s (sec / sample)" % ((time.time() - start_time) / (len(y) - 1)))
        y = decode_mu_law(y, mu=config.n_quantize)

        # save as .wav file
        write(args.outdir + "/%s.wav" % utt_id,
              args.fs,
              (y * np.iinfo(np.int16).max).astype(np.int16))


if __name__ == "__main__":
    main()

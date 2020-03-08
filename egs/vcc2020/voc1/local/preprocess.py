#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Nagoya University (Wen-Chin Huang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Perform preprocessing and raw feature extraction.
   This is slightly different from the original version,
   because we want to train 16kHz-24kHz PWG, 
   so we skip sampling rate check.
"""


import argparse
import logging
import os

import librosa
import numpy as np
import soundfile as sf
import yaml

from tqdm import tqdm

from parallel_wavegan.datasets import AudioDataset
from parallel_wavegan.datasets import AudioSCPDataset
from parallel_wavegan.utils import write_hdf5


def logmelfilterbank(audio,
                     sampling_rate,
                     fft_size=1024,
                     hop_size=256,
                     win_length=None,
                     window="hann",
                     num_mels=80,
                     fmin=None,
                     fmax=None,
                     eps=1e-10,
                     ):
    """Compute log-Mel filterbank feature.

    Args:
        audio (ndarray): Audio signal (T,).
        sampling_rate (int): Sampling rate.
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length. If set to None, it will be the same as fft_size.
        window (str): Window function type.
        num_mels (int): Number of mel basis.
        fmin (int): Minimum frequency in mel basis calculation.
        fmax (int): Maximum frequency in mel basis calculation.
        eps (float): Epsilon value to avoid inf in log calculation.

    Returns:
        ndarray: Log Mel filterbank feature (#frames, num_mels).

    """
    # get amplitude spectrogram
    x_stft = librosa.stft(audio, n_fft=fft_size, hop_length=hop_size,
                          win_length=win_length, window=window, pad_mode="reflect")
    spc = np.abs(x_stft).T  # (#frames, #bins)

    # get mel basis
    fmin = 0 if fmin is None else fmin
    fmax = sampling_rate / 2 if fmax is None else fmax
    mel_basis = librosa.filters.mel(sampling_rate, fft_size, num_mels, fmin, fmax)

    return np.log10(np.maximum(eps, np.dot(spc, mel_basis.T)))


def main():
    """Run preprocessing process."""
    parser = argparse.ArgumentParser(
        description="Preprocess audio and then extract features (See detail in parallel_wavegan/bin/preprocess.py).")
    parser.add_argument("--wav-scp", "--scp", default=None, type=str,
                        help="kaldi-style wav.scp file. you need to specify either scp or rootdir.")
    parser.add_argument("--segments", default=None, type=str,
                        help="kaldi-style segments file. if use, you must to specify both scp and segments.")
    parser.add_argument("--rootdir", default=None, type=str,
                        help="directory including wav files. you need to specify either scp or rootdir.")
    parser.add_argument("--dumpdir", type=str, required=True,
                        help="directory to dump feature files.")
    parser.add_argument("--config", type=str, required=True,
                        help="yaml format configuration file.")
    parser.add_argument("--verbose", type=int, default=1,
                        help="logging level. higher is more logging. (default=1)")
    args = parser.parse_args()

    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    else:
        logging.basicConfig(
            level=logging.WARN, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
        logging.warning('Skip DEBUG/INFO messages')

    # load config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))

    # check arguments
    if (args.wav_scp is not None and args.rootdir is not None) or \
            (args.wav_scp is None and args.rootdir is None):
        raise ValueError("Please specify either --rootdir or --wav-scp.")

    # get dataset
    if args.wav_scp is None:
        dataset = AudioDataset(
            args.rootdir, "*.wav",
            audio_load_fn=sf.read,
            return_utt_id=True,
        )
    else:
        dataset = AudioSCPDataset(
            args.wav_scp,
            segments=args.segments,
            return_utt_id=True,
            return_sampling_rate=True,
        )

    # check directly existence
    if not os.path.exists(args.dumpdir):
        os.makedirs(args.dumpdir, exist_ok=True)

    # process each data
    for utt_id, (audio, fs) in tqdm(dataset):
        # check
        assert len(audio.shape) == 1, \
            f"{utt_id} seems to be multi-channel signal."
        assert np.abs(audio).max() <= 1.0, \
            f"{utt_id} seems to be different from 16 bit PCM."

        # trim silence
        if config["trim_silence"]:
            audio, _ = librosa.effects.trim(audio,
                                            top_db=config["trim_threshold_in_db"],
                                            frame_length=config["trim_frame_size"],
                                            hop_length=config["trim_hop_size"])

        # extract feature
        mel = logmelfilterbank(audio,
                               sampling_rate=config["sampling_rate"],
                               fft_size=config["fft_size"],
                               hop_size=config["hop_size"],
                               win_length=config["win_length"],
                               window=config["window"],
                               num_mels=config["num_mels"],
                               fmin=config["fmin"],
                               fmax=config["fmax"])

        # make sure the audio length and feature length are matched
        audio = np.pad(audio, (0, config["fft_size"]), mode="edge")
        audio = audio[:len(mel) * config["hop_size"]]
        assert len(mel) * config["hop_size"] == len(audio)

        # apply global gain
        if config["global_gain_scale"] > 0.0:
            audio *= config["global_gain_scale"]
        if np.abs(audio).max() >= 1.0:
            logging.warn(f"{utt_id} causes clipping. "
                         f"it is better to re-consider global gain scale.")
            continue

        # save
        if config["format"] == "hdf5":
            write_hdf5(os.path.join(args.dumpdir, f"{utt_id}.h5"), "wave", audio.astype(np.float32))
            write_hdf5(os.path.join(args.dumpdir, f"{utt_id}.h5"), "feats", mel.astype(np.float32))
        elif config["format"] == "npy":
            np.save(os.path.join(args.dumpdir, f"{utt_id}-wave.npy"),
                    audio.astype(np.float32), allow_pickle=False)
            np.save(os.path.join(args.dumpdir, f"{utt_id}-feats.npy"),
                    mel.astype(np.float32), allow_pickle=False)
        else:
            raise ValueError("support only hdf5 or npy format.")


if __name__ == "__main__":
    main()

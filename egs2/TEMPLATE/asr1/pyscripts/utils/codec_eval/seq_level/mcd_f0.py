#!/usr/bin/env python3

# Copyright 2021 Wen-Chin Huang and Tomoki Hayashi
# Copyright 2023 Wen-Chin Huang
# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Metric calculation for codecs."""

import logging
import os
from typing import Dict, List, Tuple

import argparse
import fnmatch
import soundfile as sf

import numpy as np
import pysptk
import pyworld as pw
import scipy
from scipy.io import wavfile
from scipy.signal import firwin
from scipy.signal import lfilter
from torch._C import ErrorReport

from fastdtw import fastdtw
import librosa


MCEP_DIM=39
MCEP_ALPHA=0.466
MCEP_SHIFT=5
MCEP_FFTL=1024


def low_cut_filter(x, fs, cutoff=70):
    """Function to apply low cut filter

    Args:
        x (ndarray): Waveform sequence
        fs (int): Sampling frequency
        cutoff (float): Cutoff frequency of low cut filter

    Return:
        (ndarray): Low cut filtered waveform sequence
    """

    nyquist = fs // 2
    norm_cutoff = cutoff / nyquist

    # low cut filter
    fil = firwin(255, norm_cutoff, pass_zero=False)
    lcf_x = lfilter(fil, 1, x)

    return lcf_x


def spc2npow(spectrogram):
    """Calculate normalized power sequence from spectrogram

    Parameters
    ----------
    spectrogram : array, shape (T, `fftlen / 2 + 1`)
        Array of spectrum envelope

    Return
    ------
    npow : array, shape (`T`, `1`)
        Normalized power sequence

    """

    # frame based processing
    npow = np.apply_along_axis(_spvec2pow, 1, spectrogram)

    meanpow = np.mean(npow)
    npow = 10.0 * np.log10(npow / meanpow)

    return npow


def _spvec2pow(specvec):
    """Convert a spectrum envelope into a power

    Parameters
    ----------
    specvec : vector, shape (`fftlen / 2 + 1`)
        Vector of specturm envelope |H(w)|^2

    Return
    ------
    power : scala,
        Power of a frame

    """

    # set FFT length
    fftl2 = len(specvec) - 1
    fftl = fftl2 * 2

    # specvec is not amplitude spectral |H(w)| but power spectral |H(w)|^2
    power = specvec[0] + specvec[fftl2]
    for k in range(1, fftl2):
        power += 2.0 * specvec[k]
    power /= fftl

    return power


def extfrm(data, npow, power_threshold=-20):
    """Extract frame over the power threshold

    Parameters
    ----------
    data: array, shape (`T`, `dim`)
        Array of input data
    npow : array, shape (`T`)
        Vector of normalized power sequence.
    power_threshold : float, optional
        Value of power threshold [dB]
        Default set to -20

    Returns
    -------
    data: array, shape (`T_ext`, `dim`)
        Remaining data after extracting frame
        `T_ext` <= `T`

    """

    T = data.shape[0]
    if T != len(npow):
        raise("Length of two vectors is different.")

    valid_index = np.where(npow > power_threshold)
    extdata = data[valid_index]
    assert extdata.shape[0] <= T

    return extdata


def world_extract(x, fs, f0min, f0max):
    # scale from [-1, 1] to [-32768, 32767]
    x = x * np.iinfo(np.int16).max
    
    x = np.array(x, dtype=np.float64)
    x = low_cut_filter(x, fs)

    # extract features
    f0, time_axis = pw.harvest(
        x, fs, f0_floor=f0min, f0_ceil=f0max, frame_period=MCEP_SHIFT
    )
    sp = pw.cheaptrick(x, f0, time_axis, fs, fft_size=MCEP_FFTL)
    ap = pw.d4c(x, f0, time_axis, fs, fft_size=MCEP_FFTL)
    mcep = pysptk.sp2mc(sp, MCEP_DIM, MCEP_ALPHA)
    npow = spc2npow(sp)

    return {
        "sp": sp,
        "mcep": mcep,
        "ap": ap,
        "f0": f0,
        "npow": npow,
    }

def metrics(pred_x, gt_x, fs, f0min, f0max, dtw=False):

    pred_feats = world_extract(pred_x, fs, f0min, f0max)
    gt_feats = world_extract(gt_x, fs, f0min, f0max)

    if dtw:
        # VAD & DTW based on power
        pred_mcep_nonsil_pow = extfrm(pred_feats["mcep"], pred_feats["npow"])
        gt_mcep_nonsil_pow = extfrm(gt_feats["mcep"], gt_feats["npow"])
        _, path = fastdtw(pred_mcep_nonsil_pow, gt_mcep_nonsil_pow, dist=scipy.spatial.distance.euclidean)
        twf_pow = np.array(path).T

        # MCD using power-based DTW
        pred_mcep_dtw_pow = pred_mcep_nonsil_pow[twf_pow[0]]
        gt_mcep_dtw_pow = gt_mcep_nonsil_pow[twf_pow[1]]
        diff2sum = np.sum((pred_mcep_dtw_pow - gt_mcep_dtw_pow) ** 2, 1)
        mcd = np.mean(10.0 / np.log(10.0) * np.sqrt(2 * diff2sum), 0)

        # VAD & DTW based on f0
        gt_nonsil_f0_idx = np.where(gt_feats["f0"] > 0)[0]
        pred_nonsil_f0_idx = np.where(pred_feats["f0"] > 0)[0]
        try:
            gt_mcep_nonsil_f0 = gt_feats["mcep"][gt_nonsil_f0_idx]
            pred_mcep_nonsil_f0 = pred_feats["mcep"][pred_nonsil_f0_idx]
            _, path = fastdtw(pred_mcep_nonsil_f0, gt_mcep_nonsil_f0, dist=scipy.spatial.distance.euclidean)
            twf_f0 = np.array(path).T

            # f0RMSE, f0CORR using f0-based DTW
            pred_f0_dtw = pred_feats["f0"][pred_nonsil_f0_idx][twf_f0[0]]
            gt_f0_dtw = gt_feats["f0"][gt_nonsil_f0_idx][twf_f0[1]]
            f0rmse = np.sqrt(np.mean((pred_f0_dtw - gt_f0_dtw) ** 2))
            f0corr = scipy.stats.pearsonr(pred_f0_dtw, gt_f0_dtw)[0]
        except ValueError:
            logging.warning(
                "No nonzero f0 is found. Skip f0rmse f0corr computation and set them to NaN. "
                "This might due to unconverge training. Please tune the training time and hypers."
            )
            f0rmse = np.nan
            f0corr = np.nan
        
    else:
        diff2sum = np.sum((pred_feats["mcep"] - gt_feats["mcep"]) ** 2, 1)
        mcd = np.mean(10 / np.log(10.0) * np.sqrt(2 * diff2sum), 0)
        f0rmse = np.sqrt(np.mean((pred_feats["f0"] - gt_feats["f0"]) ** 2))
        f0corr = scipy.stats.pearsonr(pred_feats["f0"], gt_feats["f0"])[0]
    
    return {
        "mcd": mcd,
        "f0rmse": f0rmse,
        "f0corr": f0corr,
    }


def find_files(
    root_dir: str, query: List[str] = ["*.flac", "*.wav"], include_root_dir: bool = True
) -> List[str]:
    """Find files recursively.

    Args:
        root_dir (str): Root root_dir to find.
        query (List[str]): Query to find.
        include_root_dir (bool): If False, root_dir name is not included.

    Returns:
        List[str]: List of found filenames.

    """
    files = []
    for root, dirnames, filenames in os.walk(root_dir, followlinks=True):
        for q in query:
            for filename in fnmatch.filter(filenames, q):
                files.append(os.path.join(root, filename))
    if not include_root_dir:
        files = [file_.replace(root_dir + "/", "") for file_ in files]

    return files


def _get_basename(path: str) -> str:
    return os.path.splitext(os.path.split(path)[-1])[0]


def get_parser() -> argparse.Namespace:
    """Get argument parser."""
    parser = argparse.ArgumentParser(
        description="Evaluate Speech Resynthesis Performance."
    )
    parser.add_argument(
        "gen_wavdir_or_wavscp",
        type=str,
        help="Path of directory or wav.scp for generated waveforms.",
    )
    parser.add_argument(
        "gt_wavdir_or_wavscp",
        type=str,
        help="Path of directory or wav.scp for ground truth waveforms.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        help="Path of directory to write the results.",
    )
    parser.add_argument(
        "--fs",
        type=humanfriendly_or_none,
        default=None,
        help="If the sampling rate specified, Change the sampling rate.",
    )
    parser.add_argument(
        "--f0min",
        default=0,
        type=int,
        help="Minimum f0 value.",
    )
    parser.add_argument(
        "--f0max",
        default=8000,
        type=int,
        help="Maximum f0 value.",
    )
    parser.add_argument(
        "--dtw",
        default=False,
        type=bool,
        help="Whether to apply dtw for the calculation."
    )
    parser.add_argument(
        "--verbose",
        default=1,
        type=int,
        help="Verbosity level. Higher is more logging.",
    )
    return parser


def main():
    """Run Speech Resynthesis Evaluation."""
    args = get_parser().parse_args()

    # logging info
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    # find files
    if os.path.isdir(args.gen_wavdir_or_wavscp):
        gen_files = sorted(find_files(args.gen_wavdir_or_wavscp))
    else:
        with open(args.gen_wavdir_or_wavscp) as f:
            gen_files = [line.strip().split(None, 1)[1] for line in f.readlines()]
        if gen_files[0].endswith("|"):
            raise ValueError("Not supported wav.scp format.")
    if os.path.isdir(args.gt_wavdir_or_wavscp):
        gt_files = sorted(find_files(args.gt_wavdir_or_wavscp))
    else:
        with open(args.gt_wavdir_or_wavscp) as f:
            gt_files = [line.strip().split(None, 1)[1] for line in f.readlines()]
        if gt_files[0].endswith("|"):
            raise ValueError("Not supported wav.scp format.")

    if len(gen_files) == 0:
        raise FileNotFoundError("Not found any generated audio files.")
    if len(gen_files) > len(gt_files):
        raise ValueError(
            "#groundtruth files are less than #generated files "
            f"(#gen={len(gen_files)} vs. #gt={len(gt_files)}). "
            "Please check the groundtruth directory."
        )
    logging.info("The number of utterances = %d" % len(gen_files))

    score_dict = {}
    for (gen_f, gt_f) in zip(gen_files, gt_files):
        pred_x, pred_fs = sf.read(gen_f, dtype="int16")
        gt_x, gt_fs = sf.read(gt_f, dtype="int16")

        if pred_fs != fs:
            pred_x = librosa.resample(pred_x.astype(np.float), pred_fs, args.fs)
        if gt_fs != fs:
            gt_x = librosa.resample(gt_x.astype(np.float), gt_fs, args.fs)
        
        scores = metrics(pred_x, gt_x, args.fs, args.f0min, args.f0max, dtw=False)
        gt_basename = _get_basename(gt_f)
        score_dict[gt_basename] = scores

    # write results
    if args.outdir is None:
        if os.path.isdir(args.gen_wavdir_or_wavscp):
            args.outdir = args.gen_wavdir_or_wavscp
        else:
            args.outdir = os.path.dirname(args.gen_wavdir_or_wavscp)
    os.makedirs(args.outdir, exist_ok=True)

    for metric in ["mcd", "f0rmse", "f0corr"]:
        with open(f"{args.outdir}/utt2{}".format(metric), "w") as f:
            for utt_id in sorted(score_dict.keys()):
                score = score_dict[utt_id][metric]
                f.write(f"{utt_id} {metric:.4f}\n")
        logging.info("Successfully finished {} evaluation.".format(metric))
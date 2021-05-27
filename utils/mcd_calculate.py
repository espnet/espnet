#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2020 Nagoya University (Wen-Chin Huang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Calculate MCD using converted waveform.

import argparse
import fnmatch
import multiprocessing as mp
import os

from fastdtw import fastdtw
import numpy as np
import pysptk
import pyworld as pw
import scipy
from scipy.io import wavfile
from scipy.signal import firwin
from scipy.signal import lfilter


def find_files(root_dir, query="*.wav", include_root_dir=True):
    """Find files recursively.

    Args:
        root_dir (str): Root root_dir to find.
        query (str): Query to find.
        include_root_dir (bool): If False, root_dir name is not included.

    Returns:
        list: List of found filenames.

    """
    files = []
    for root, dirnames, filenames in os.walk(root_dir, followlinks=True):
        for filename in fnmatch.filter(filenames, query):
            files.append(os.path.join(root, filename))
    if not include_root_dir:
        files = [file_.replace(root_dir + "/", "") for file_ in files]

    return files


def low_cut_filter(x, fs, cutoff=70):
    """FUNCTION TO APPLY LOW CUT FILTER

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
        raise ("Length of two vectors is different.")

    valid_index = np.where(npow > power_threshold)
    extdata = data[valid_index]
    assert extdata.shape[0] <= T

    return extdata


def world_extract(wav_path, args):
    fs, x = wavfile.read(wav_path)
    x = np.array(x, dtype=np.float64)
    x = low_cut_filter(x, fs)

    # extract features
    f0, time_axis = pw.harvest(
        x, fs, f0_floor=args.f0min, f0_ceil=args.f0max, frame_period=args.shiftms
    )
    sp = pw.cheaptrick(x, f0, time_axis, fs, fft_size=args.fftl)
    ap = pw.d4c(x, f0, time_axis, fs, fft_size=args.fftl)
    mcep = pysptk.sp2mc(sp, args.mcep_dim, args.mcep_alpha)
    npow = spc2npow(sp)

    return {
        "sp": sp,
        "mcep": mcep,
        "ap": ap,
        "f0": f0,
        "npow": npow,
    }


def get_basename(path):
    return os.path.splitext(os.path.split(path)[-1])[0]


def calculate(file_list, gt_file_list, args, MCD):

    for i, cvt_path in enumerate(file_list):
        corresponding_list = list(
            filter(lambda gt_path: get_basename(gt_path) in cvt_path, gt_file_list)
        )
        assert len(corresponding_list) == 1
        gt_path = corresponding_list[0]
        gt_basename = get_basename(gt_path)

        # extract ground truth and converted features
        gt_feats = world_extract(gt_path, args)
        cvt_feats = world_extract(cvt_path, args)

        # VAD & DTW based on power
        gt_mcep_nonsil_pow = extfrm(gt_feats["mcep"], gt_feats["npow"])
        cvt_mcep_nonsil_pow = extfrm(cvt_feats["mcep"], cvt_feats["npow"])
        _, path = fastdtw(
            cvt_mcep_nonsil_pow,
            gt_mcep_nonsil_pow,
            dist=scipy.spatial.distance.euclidean,
        )
        twf_pow = np.array(path).T

        # MCD using power-based DTW
        cvt_mcep_dtw_pow = cvt_mcep_nonsil_pow[twf_pow[0]]
        gt_mcep_dtw_pow = gt_mcep_nonsil_pow[twf_pow[1]]
        diff2sum = np.sum((cvt_mcep_dtw_pow - gt_mcep_dtw_pow) ** 2, 1)
        mcd = np.mean(10.0 / np.log(10.0) * np.sqrt(2 * diff2sum), 0)

        print("{} {}".format(gt_basename, mcd))
        MCD.append(mcd)


def get_parser():

    parser = argparse.ArgumentParser(description="calculate MCD.")
    parser.add_argument(
        "--wavdir",
        required=True,
        type=str,
        help="path of directory for converted waveforms",
    )
    parser.add_argument(
        "--gtwavdir",
        required=True,
        type=str,
        help="path of directory for ground truth waveforms",
    )

    # analysis related
    parser.add_argument(
        "--mcep_dim", default=41, type=int, help="dimension of mel cepstrum coefficient"
    )
    parser.add_argument(
        "--mcep_alpha", default=0.41, type=int, help="all pass constant"
    )
    parser.add_argument("--fftl", default=1024, type=int, help="fft length")
    parser.add_argument("--shiftms", default=5, type=int, help="frame shift (ms)")
    parser.add_argument(
        "--f0min", required=True, type=int, help="fo search range (min)"
    )
    parser.add_argument(
        "--f0max", required=True, type=int, help="fo search range (max)"
    )

    parser.add_argument(
        "--n_jobs", default=40, type=int, help="number of parallel jobs"
    )
    return parser


def main():
    args = get_parser().parse_args()

    # find files
    converted_files = sorted(find_files(args.wavdir))
    gt_files = sorted(find_files(args.gtwavdir))

    # Get and divide list

    print("number of utterances = %d" % len(converted_files))
    file_lists = np.array_split(converted_files, args.n_jobs)
    file_lists = [f_list.tolist() for f_list in file_lists]

    # multi processing
    with mp.Manager() as manager:
        MCD = manager.list()
        processes = []
        for f in file_lists:
            p = mp.Process(target=calculate, args=(f, gt_files, args, MCD))
            p.start()
            processes.append(p)

        # wait for all process
        for p in processes:
            p.join()

        mMCD = np.mean(np.array(MCD))
        print("Mean MCD: {:.2f}".format(mMCD))


if __name__ == "__main__":
    main()

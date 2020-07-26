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

    return {
        "sp": sp,
        "mcep": mcep,
        "ap": ap,
        "f0": f0,
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

        # non-silence parts
        gt_idx = np.where(gt_feats["f0"] > 0)[0]
        gt_mcep = gt_feats["mcep"][gt_idx]
        cvt_idx = np.where(cvt_feats["f0"] > 0)[0]
        cvt_mcep = cvt_feats["mcep"][cvt_idx]

        # DTW
        _, path = fastdtw(cvt_mcep, gt_mcep, dist=scipy.spatial.distance.euclidean)
        twf = np.array(path).T
        cvt_mcep_dtw = cvt_mcep[twf[0]]
        gt_mcep_dtw = gt_mcep[twf[1]]

        # MCD
        diff2sum = np.sum((cvt_mcep_dtw - gt_mcep_dtw) ** 2, 1)
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

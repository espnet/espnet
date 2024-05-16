#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Evaluate SI-SNR."""

import argparse
import fnmatch
import logging
import multiprocessing as mp
import os
from typing import Dict, List, Tuple

import torch


import librosa
import numpy as np
import soundfile as sf
from espnet2.enh.loss.criterions.time_domain import SISNRLoss


si_snr_loss = SISNRLoss()


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


def calculate(
    file_list: List[str],
    gt_file_list: List[str],
    args: argparse.Namespace,
    si_snr_score_dict: Dict,
):
    """Calculate SI-SNR."""
    for i, gen_path in enumerate(file_list):
        corresponding_list = list(
            filter(lambda gt_path: _get_basename(gt_path) in gen_path, gt_file_list)
        )
        assert len(corresponding_list) == 1
        gt_path = corresponding_list[0]
        gt_basename = _get_basename(gt_path)

        # load wav file as int16
        gen_x, gen_fs = sf.read(gen_path, dtype="int16")
        gt_x, gt_fs = sf.read(gt_path, dtype="int16")

        fs = gen_fs
        if gen_fs != gt_fs:
            gt_x = librosa.resample(gt_x.astype(np.float), gt_fs, gen_fs)

        # align size
        min_length = min(gen_x.shape[0], gt_x.shape[0])
        # assert min_length > gen_x.shape[0] * 0.95 and min_length > gt_x.shape[0] * 0.95, "significant length mismatch detected for {} with reference {} and ground truth {}".format(gt_basename, gen_x.shape[0], ref_x.shape[0])

        gen_x = gen_x[:min_length]
        gt_x = gt_x[:min_length]

        # extract ground truth and converted features
        si_snr_score = -float(
            si_snr_loss(
                torch.from_numpy(gt_x).float(),
                torch.from_numpy(gen_x).float(),
            )
        )
        logging.info(f"{gt_basename} {si_snr_score:.4f}")
        si_snr_score_dict[gt_basename] = si_snr_score


def get_parser() -> argparse.Namespace:
    """Get argument parser."""
    parser = argparse.ArgumentParser(description="Evaluate Mel-cepstrum distortion.")
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

    # analysis related
    parser.add_argument(
        "--nj",
        default=16,
        type=int,
        help="Number of parallel jobs.",
    )
    parser.add_argument(
        "--verbose",
        default=1,
        type=int,
        help="Verbosity level. Higher is more logging.",
    )
    return parser


def main():
    """Run SI-SNR calculation in parallel."""
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

    # Get and divide list
    if len(gen_files) == 0:
        raise FileNotFoundError("Not found any generated audio files.")
    if len(gen_files) > len(gt_files):
        raise ValueError(
            "#groundtruth files are less than #generated files "
            f"(#gen={len(gen_files)} vs. #gt={len(gt_files)}). "
            "Please check the groundtruth directory."
        )
    logging.info("The number of utterances = %d" % len(gen_files))
    file_lists = np.array_split(gen_files, args.nj)
    file_lists = [f_list.tolist() for f_list in file_lists]

    # multi processing
    with mp.Manager() as manager:
        si_snr_score_dict = manager.dict()
        processes = []
        for f in file_lists:
            p = mp.Process(
                target=calculate, args=(f, gt_files, args, si_snr_score_dict)
            )
            p.start()
            processes.append(p)

        # wait for all process
        for p in processes:
            p.join()

        # convert to standard list
        si_snr_score_dict = dict(si_snr_score_dict)

        # calculate statistics
        mean_si_snr_score = np.mean(np.array([v for v in si_snr_score_dict.values()]))
        std_si_snr_score = np.std(np.array([v for v in si_snr_score_dict.values()]))
        logging.info(f"Average: {mean_si_snr_score:.4f} ± {std_si_snr_score:.4f}")

    # write results
    if args.outdir is None:
        if os.path.isdir(args.gen_wavdir_or_wavscp):
            args.outdir = args.gen_wavdir_or_wavscp
        else:
            args.outdir = os.path.dirname(args.gen_wavdir_or_wavscp)
    os.makedirs(args.outdir, exist_ok=True)
    with open(f"{args.outdir}/utt2sisnr", "w") as f:
        for utt_id in sorted(si_snr_score_dict.keys()):
            si_snr_score = si_snr_score_dict[utt_id]
            f.write(f"{utt_id} {si_snr_score:.4f}\n")
    with open(f"{args.outdir}/si_snr_score_avg_result.txt", "w") as f:
        f.write(f"#utterances: {len(gen_files)}\n")
        f.write(f"Average: {mean_si_snr_score:.4f} ± {std_si_snr_score:.4f}")

    logging.info("Successfully finished SI-SNR evaluation.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

# Copyright 2024 Takaaki Saeki
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""
Evaluate SpeechBLEU between generated and groundtruth audios.
https://arxiv.org/abs/2401.16812
This can be used to evaluate the performance of text-to-speech,
voice conversion, speech enhancement, etc.

Using discrete-speech-metrics package:
https://github.com/Takaaki-Saeki/DiscreteSpeechMetrics
"""

import argparse
import fnmatch
import logging
import os
from typing import List

import librosa
import numpy as np
import soundfile as sf
import torch
from discrete_speech_metrics import SpeechBLEU
from scipy import spatial


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
        description="Evaluate SpeechBLEU (https://arxiv.org/abs/2401.16812)."
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
        "--verbose",
        default=1,
        type=int,
        help="Verbosity level. Higher is more logging.",
    )
    return parser


def main():
    """Run SpeechBLEU calculation."""
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

    # Using the best configuration of SpeechBLEU.
    SR = 16000
    metrics = SpeechBLEU(
        sr=16000,
        model_type="hubert-base",
        vocab=200,
        layer=11,
        n_ngram=2,
        remove_repetition=True,
        use_gpu=True,
    )

    # calculate SpeechBLEU.
    spbleu_dict = dict()
    for i, gen_path in enumerate(gen_files):
        corresponding_list = list(
            filter(lambda gt_path: _get_basename(gt_path) in gen_path, gt_files)
        )
        assert len(corresponding_list) == 1
        gt_path = corresponding_list[0]
        gt_basename = _get_basename(gt_path)

        # load wav file as int16
        gen_x, gen_fs = sf.read(gen_path, dtype="int16")
        gt_x, gt_fs = sf.read(gt_path, dtype="int16")

        if gt_fs != SR:
            gt_x = librosa.resample(gt_x.astype(np.float), gt_fs, SR)
        if gen_fs != SR:
            gen_x = librosa.resample(gen_x.astype(np.float), gen_fs, SR)

        # Amp Normalization -1 ~ 1
        gen_amax = np.amax(np.absolute(gen_x))
        gen_x = gen_x.astype(np.float32) / gen_amax
        gt_amax = np.amax(np.absolute(gt_x))
        gt_x = gt_x.astype(np.float32) / gt_amax
        # Calculate SpeechBLEU.
        spbleu = metrics.score(gt_x, gen_x)
        logging.info(f"{gt_basename} {spbleu:.4f}")
        spbleu_dict[gt_basename] = spbleu

    # calculate statistics
    mean_spbleu = np.mean(np.array([v for v in spbleu_dict.values()]))
    std_spbleu = np.std(np.array([v for v in spbleu_dict.values()]))
    logging.info(f"Average: {mean_spbleu:.4f} ± {std_spbleu:.4f}")

    # write results
    if args.outdir is None:
        if os.path.isdir(args.gen_wavdir_or_wavscp):
            args.outdir = args.gen_wavdir_or_wavscp
        else:
            args.outdir = os.path.dirname(args.gen_wavdir_or_wavscp)
    os.makedirs(args.outdir, exist_ok=True)
    with open(f"{args.outdir}/utt2spbleu", "w") as f:
        for utt_id in sorted(spbleu_dict.keys()):
            spbleu = spbleu_dict[utt_id]
            f.write(f"{utt_id} {spbleu:.4f}\n")
    with open(f"{args.outdir}/spbleu_avg_result.txt", "w") as f:
        f.write(f"#utterances: {len(gen_files)}\n")
        f.write(f"Average: {mean_spbleu:.4f} ± {std_spbleu:.4f}")

    logging.info("Successfully finished SpeechBLEU evaluation.")


if __name__ == "__main__":
    main()

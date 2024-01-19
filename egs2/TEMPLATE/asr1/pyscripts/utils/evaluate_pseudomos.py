#!/usr/bin/env python3

# Copyright 2023 Takaaki Saeki
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Evaluate pseudo MOS calculated by automatic MOS prediction model."""

import argparse
import fnmatch
import logging
import os
from typing import List

import librosa
import numpy as np
import torch


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
    predictor: torch.nn.Module,
    device: torch.device,
    batchsize: int,
):
    """Calculate pseudo MOS."""
    pmos_dict = {}
    fs = librosa.get_samplerate(file_list[0])
    for si, _ in enumerate(file_list[::batchsize]):
        gen_paths = []
        gen_xs = []
        for bi in range(batchsize):
            if si * batchsize + bi >= len(file_list):
                break
            gen_path = file_list[si * batchsize + bi]
            gen_paths.append(gen_path)
            gen_x, gen_fs = librosa.load(gen_path, sr=None, mono=True)
            # Assuming same sampling rate for all files.
            assert fs == gen_fs
            gen_xs.append(gen_x)
        # Padding
        max_len = max([len(gen_x) for gen_x in gen_xs])
        for bi in range(batchsize):
            if si * batchsize + bi >= len(file_list):
                break
            gen_xs[bi] = np.pad(
                gen_xs[bi],
                (0, max_len - len(gen_xs[bi])),
                "constant",
                constant_values=0,
            )
        gen_xs = torch.from_numpy(np.stack(gen_xs)).to(device)
        # Compute pseudo MOS.
        scores = predictor(gen_xs, fs)
        for gen_path, score in zip(gen_paths, scores):
            gen_basename = _get_basename(gen_path)
            pmos = score.item()
            logging.info(f"{gen_basename} {pmos:.4f}")
            pmos_dict[gen_basename] = pmos
    return pmos_dict


def get_parser() -> argparse.Namespace:
    """Get argument parser."""
    parser = argparse.ArgumentParser(description="Evaluate pseudo MOS.")
    parser.add_argument(
        "gen_wavdir_or_wavscp",
        type=str,
        help="Path of directory or wav.scp for generated waveforms.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        help="Path of directory to write the results.",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Inference device")
    parser.add_argument(
        "--mos_toolkit",
        type=str,
        default="utmos",
        choices=["utmos"],
        help="Toolkit to calculate pseudo MOS.",
    )
    parser.add_argument(
        "--batchsize",
        default=4,
        type=int,
        help="Number of batches.",
    )
    parser.add_argument(
        "--verbose",
        default=1,
        type=int,
        help="Verbosity level. Higher is more logging.",
    )
    return parser


def main():
    """Run pseudo MOS calculation in parallel."""
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

    # Find files for generated speech waveforms.
    if os.path.isdir(args.gen_wavdir_or_wavscp):
        gen_files = sorted(find_files(args.gen_wavdir_or_wavscp))
    else:
        with open(args.gen_wavdir_or_wavscp) as f:
            gen_files = [line.strip().split(None, 1)[1] for line in f.readlines()]
        if gen_files[0].endswith("|"):
            raise ValueError("Not supported wav.scp format.")

    # Get and divide list
    if len(gen_files) == 0:
        raise FileNotFoundError("Not found any generated audio files.")
    logging.info("The number of utterances = %d" % len(gen_files))

    if torch.cuda.is_available() and ("cuda" in args.device):
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")

    if args.mos_toolkit == "utmos":
        # Load predictor for UTMOS22.
        predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong").to(
            device
        )
    else:
        raise NotImplementedError(f"Not supported {args.mos_toolkit}.")

    # Calculate pseudo MOS for all the files.
    pmos_dict = calculate(gen_files, predictor, device, args.batchsize)

    # convert to standard list
    pmos_dict = dict(pmos_dict)

    # calculate statistics
    mean_pmos = np.mean(np.array([v for v in pmos_dict.values()]))
    std_pmos = np.std(np.array([v for v in pmos_dict.values()]))
    logging.info(f"Average: {mean_pmos:.4f} ± {std_pmos:.4f}")

    # Write results
    if args.outdir is None:
        if os.path.isdir(args.gen_wavdir_or_wavscp):
            args.outdir = args.gen_wavdir_or_wavscp
        else:
            args.outdir = os.path.dirname(args.gen_wavdir_or_wavscp)
    os.makedirs(args.outdir, exist_ok=True)
    with open(f"{args.outdir}/utt2pmos", "w") as f:
        for utt_id in sorted(pmos_dict.keys()):
            pmos = pmos_dict[utt_id]
            f.write(f"{utt_id} {pmos:.4f}\n")
    with open(f"{args.outdir}/pmos_avg_result.txt", "w") as f:
        f.write(f"#utterances: {len(gen_files)}\n")
        f.write(f"Average: {mean_pmos:.4f} ± {std_pmos:.4f}")

    logging.info("Successfully finished pseudo MOS evaluation.")


if __name__ == "__main__":
    main()

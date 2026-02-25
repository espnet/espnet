#!/usr/bin/env python3

# Copyright 2020 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Trim silence in the audio and re-create wav files."""

import argparse
import fnmatch
import logging
import os
import sys

import numpy as np
import soundfile as sf


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


def main():
    """Run trimming."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_sp_length",
        default=0.05,
        type=float,
        help="maximum length of shortpose in the middle of utterances",
    )
    parser.add_argument(
        "--after_sp_length",
        default=0.05,
        type=float,
        help="length of offset after shortpose",
    )
    parser.add_argument("db_root", type=str, help="root path of NKY corpus")
    parser.add_argument(
        "target_dir", type=str, help="directory to save the trimmed audio"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG,
        stream=sys.stdout,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    db_root = args.db_root
    target_dir = args.target_dir
    max_sp_length = args.max_sp_length
    after_sp_length = args.after_sp_length
    wavfiles = sorted(find_files(db_root, include_root_dir=False))
    labfiles = [f.replace(".wav", ".lab") for f in wavfiles]

    for idx, (wavfile, labfile) in enumerate(zip(wavfiles, labfiles)):
        if not os.path.exists(f"{db_root}/{labfile}"):
            logging.warning(f"{labfile} does not exist. skipped.")
            continue
        x, fs = sf.read(f"{db_root}/{wavfile}")
        with open(f"{db_root}/{labfile}") as f:
            lines = [line.replace("\n", "") for line in f.readlines()]
        start_times = [float(line.split("\t")[0]) for line in lines]
        end_times = [float(line.split("\t")[1]) for line in lines]
        labels = [line.split("\t")[2] for line in lines]
        start_idxs = [int(t * fs) for t in start_times]
        end_idxs = [int(t * fs) for t in end_times]

        new_x = []
        prev_label = None
        prev_sp_legnth = 0.0
        for start_idx, end_idx, label in zip(start_idxs, end_idxs, labels):
            if label == "sil" or len(label) == 0:
                continue
            elif label in ["sp", "spn"]:
                sp_length = (end_idx - start_idx) / fs
                if sp_length > max_sp_length:
                    end_idx = start_idx + int(max_sp_length * fs)
                prev_sp_legnth = sp_length
            elif prev_label in ["sp", "spn"]:
                if prev_sp_legnth > after_sp_length + max_sp_length:
                    start_idx -= int(after_sp_length * fs)
            new_x += [x[start_idx:end_idx]]
            prev_label = label
        new_x = np.concatenate(new_x).reshape(-1)
        write_path = f"{target_dir}/{wavfile}"
        os.makedirs(os.path.dirname(write_path), exist_ok=True)
        sf.write(f"{target_dir}/{wavfile}", new_x, fs)

        if (idx + 1) % 1000 == 0:
            logging.info(f"Now processing... ({idx + 1}/{len(wavfiles)})")


if __name__ == "__main__":
    main()

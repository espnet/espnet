import json
import os
import random
import sys

import numpy as np
import soundfile as sf
from tqdm import tqdm

DATA_READ_ROOT = sys.argv[1]
DATA_WRITE_ROOT = sys.argv[2]


def read_data_file(filename,skip_lt_10s=False):
    data = []
    if "unbalanced_train_segments" in filename:
        wav_directory = "unbalanced_wav"
    elif "balanced_train_segments" in filename:
        wav_directory = "balance_wav"
    elif "eval_segments" in filename:
        wav_directory = "eval_wav"
    else:
        raise ValueError("Unknown data file")
    with open(filename, "r") as file:
        lines = file.readlines()
        for line in tqdm(lines, desc="Reading data files"):
            line = line.strip()
            if line.startswith("#"):
                continue
            parts = line.strip().split(",", maxsplit=3)
            # Extract the fields
            yt_id = parts[0]
            start_seconds = float(parts[1])
            end_seconds = float(parts[2])
            if skip_lt_10s and end_seconds - start_seconds < 10:
                continue
            data.append(
                {
                    "yt_id": yt_id,
                    "start_seconds": start_seconds,
                    "end_seconds": end_seconds,
                    "wav_directory": wav_directory,
                }
            )
    return data


eval_set = read_data_file(os.path.join(DATA_READ_ROOT, "eval_segments.csv"))
train_set = read_data_file(
    os.path.join(DATA_READ_ROOT, "unbalanced_train_segments.csv")
)
train_set_bal = read_data_file(
    os.path.join(DATA_READ_ROOT, "balanced_train_segments.csv")
)
train_set = train_set + train_set_bal

print(f"Train set size: {len(train_set)}")
print(f"Eval set size: {len(eval_set)}")

for dataset, name in [(train_set, "train"), (eval_set, "eval")]:
    missing_wav_file = 0
    wav_scp_write_path = os.path.join(DATA_WRITE_ROOT, name, "wav.scp")
    utt2spk_write_path = os.path.join(DATA_WRITE_ROOT, name, "utt2spk")

    os.makedirs(os.path.dirname(wav_scp_write_path), exist_ok=True)
    os.makedirs(os.path.dirname(utt2spk_write_path), exist_ok=True)
    with open(wav_scp_write_path, "w") as wav_f, open(
        utt2spk_write_path, "w"
    ) as utt2spk_f:
        for uttid, item in enumerate(tqdm(dataset, desc=f"Processing {name} set")):
            wav_directory = item["wav_directory"]
            wav_path = os.path.join(
                DATA_READ_ROOT, wav_directory, item["yt_id"] + ".wav"
            )
            if not os.path.exists(wav_path):
                missing_wav_file += 1
                continue
            print(f"as20k-{name}-{uttid} {wav_path}", file=wav_f)
            print(f"as20k-{name}-{uttid} dummy", file=utt2spk_f)
    print(f"Missing {missing_wav_file} wav files in {name} set.")

import json
import os
import random
import sys

import numpy as np
import soundfile as sf
from tqdm import tqdm

DATA_READ_ROOT = sys.argv[1]
DATA_WRITE_ROOT = sys.argv[2]


def read_data_file(filename, skip_lt_10s=False):
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
        missing_wav_file = 0
        skipped_wav_files = 0
        for line in tqdm(lines, desc="Reading data files"):
            line = line.strip()
            if line.startswith("#"):
                continue
            parts = line.strip().split(",", maxsplit=3)
            # Extract the fields
            yt_id = parts[0]
            start_seconds = float(parts[1])
            end_seconds = float(parts[2])
            wav_path = os.path.join(DATA_READ_ROOT, wav_directory, yt_id + ".wav")
            audio_len = end_seconds - start_seconds
            if not os.path.exists(wav_path):
                missing_wav_file += 1
                continue
            if audio_len < 10:
                if skip_lt_10s:
                    skipped_wav_files += 1
                    continue
                s, r = sf.read(
                    os.path.join(DATA_READ_ROOT, wav_directory, yt_id + ".wav")
                )
                wav_path = os.path.join(DATA_READ_ROOT, "cut_wav", yt_id + ".wav")
                os.makedirs(os.path.dirname(wav_path), exist_ok=True)
                sf.write(wav_path, s[: int(r * audio_len)], r)
            data.append(
                {
                    "wav_path": wav_path,
                    "audio_len": audio_len,
                }
            )
        print(f"Missing wav files: {missing_wav_file}, skipped: {skipped_wav_files}")
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

for dataset, name in [(train_set, "AudioSet"), (eval_set, "eval")]:
    missing_wav_file = 0
    wav_scp_write_path = os.path.join(DATA_WRITE_ROOT, name, "wav.scp")
    utt2spk_write_path = os.path.join(DATA_WRITE_ROOT, name, "utt2spk")

    os.makedirs(os.path.dirname(wav_scp_write_path), exist_ok=True)
    os.makedirs(os.path.dirname(utt2spk_write_path), exist_ok=True)
    with open(wav_scp_write_path, "w") as wav_f, open(
        utt2spk_write_path, "w"
    ) as utt2spk_f:
        for uttid, item in enumerate(tqdm(dataset, desc=f"Processing {name} set")):
            wav_path = item["wav_path"]
            print(f"as2m_20k-{name}-{uttid} {wav_path}", file=wav_f)
            print(f"as2m_20k-{name}-{uttid} dummy", file=utt2spk_f)

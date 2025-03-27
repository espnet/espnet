import json
import os
import random
import sys

import soundfile as sf
from tqdm import tqdm

DATA_READ_ROOT = sys.argv[1]
DATA_WRITE_ROOT = sys.argv[2]


def read_data_file(filename, mid2name):
    data = []
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
            labels = parts[3].replace('"', "").split(",")
            labels = [mid2name[label.strip()] for label in labels]
            data.append(
                {
                    "yt_id": yt_id,
                    "start_seconds": start_seconds,
                    "end_seconds": end_seconds,
                    "labels": labels,
                }
            )
    return data


def read_mid2name_map(mid2name_file):
    mid2name = {}
    unique_names = set()
    with open(mid2name_file, "r") as file:
        lines = file.readlines()
        for line in tqdm(lines, desc="Reading mid2name map"):
            line = line.strip()
            if line.startswith("#"):
                continue
            parts = line.strip().split(",", maxsplit=2)
            label_name = parts[2].replace('"', "").strip().replace(" ", "_")
            if label_name in unique_names:
                print(f"Warning: Duplicate label name {label_name}")
            unique_names.add(label_name)
            mid2name[parts[1]] = label_name
    return mid2name


mid2name = read_mid2name_map(os.path.join(DATA_READ_ROOT, "class_labels_indices.csv"))
eval_set = read_data_file(os.path.join(DATA_READ_ROOT, "eval_segments.csv"), mid2name)
train_set = read_data_file(
    os.path.join(DATA_READ_ROOT, "balanced_train_segments.csv"), mid2name
)

"""
Create validation split from eval
Since the code for BEATs evals is not public, it is hard to estimate
how they create val set. However, it seems like AST is using all of the
training data. To replicate this setup we do not use remove any data
from train set and use 10% of eval set as val set. This results in ~1%
gain in mAP score. AST- https://github.com/YuanGongND/ast/tree/master
"""

total_len = len(eval_set)
val_len = total_len // 10

random.seed(0)

val_set = eval_set[:val_len]

print(f"Train set size: {len(train_set)}")
print(f"Val set size: {len(val_set)}")
print(f"Eval set size: {len(eval_set)}")

for dataset, name in [(train_set, "train"), (val_set, "val"), (eval_set, "eval")]:
    missing_wav_file = 0
    text_write_path = os.path.join(DATA_WRITE_ROOT, name, "text")
    wav_scp_write_path = os.path.join(DATA_WRITE_ROOT, name, "wav.scp")
    utt2spk_write_path = os.path.join(DATA_WRITE_ROOT, name, "utt2spk")

    os.makedirs(os.path.dirname(text_write_path), exist_ok=True)
    os.makedirs(os.path.dirname(wav_scp_write_path), exist_ok=True)
    os.makedirs(os.path.dirname(utt2spk_write_path), exist_ok=True)

    with open(text_write_path, "w") as text_f, open(
        wav_scp_write_path, "w"
    ) as wav_f, open(utt2spk_write_path, "w") as utt2spk_f:
        for uttid, item in enumerate(tqdm(dataset, desc=f"Processing {name} set")):
            wav_directory = "balance_wav" if name == "train" else "eval_wav"
            wav_path = os.path.join(
                DATA_READ_ROOT, wav_directory, item["yt_id"] + ".wav"
            )
            if not os.path.exists(wav_path):
                missing_wav_file += 1
                continue
            text = " ".join(item["labels"])
            print(f"as20k-{name}-{uttid} {text}", file=text_f)
            print(f"as20k-{name}-{uttid} {wav_path}", file=wav_f)
            print(f"as20k-{name}-{uttid} dummy", file=utt2spk_f)
    print(f"Missing {missing_wav_file} wav files in {name} set.")

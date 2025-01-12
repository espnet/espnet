import json
import os
import random
import sys
import numpy as np
import soundfile as sf
from tqdm import tqdm
import numpy as np

DATA_READ_ROOT = sys.argv[1]
DATA_WRITE_ROOT = sys.argv[2]


def read_data_file(filename, mid2name):
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
            labels = parts[3].replace('"', "").split(",")
            labels = [mid2name[label.strip()] for label in labels]
            data.append(
                {
                    "yt_id": yt_id,
                    "start_seconds": start_seconds,
                    "end_seconds": end_seconds,
                    "labels": labels,
                    "wav_directory": wav_directory,
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


def generate_sampling_weights(train_set, mid2name):
    label_count = {name: 0 for name in mid2name.values()}
    for item in train_set:
        for label in item["labels"]:
            label_count[label] += 1
    label_weight = {
        name: 1000.0 / (count + 0.01) for name, count in label_count.items()
    }
    instance_wise_sampling_weights = np.zeros(len(train_set))
    for idx, item in enumerate(train_set):
        for label in item["labels"]:
            instance_wise_sampling_weights[idx] += label_weight[label]
    return instance_wise_sampling_weights


mid2name = read_mid2name_map(os.path.join(DATA_READ_ROOT, "class_labels_indices.csv"))
eval_set = read_data_file(os.path.join(DATA_READ_ROOT, "eval_segments.csv"), mid2name)
train_set = read_data_file(
    os.path.join(DATA_READ_ROOT, "unbalanced_train_segments.csv"), mid2name
)
train_set_bal = read_data_file(
    os.path.join(DATA_READ_ROOT, "balanced_train_segments.csv"), mid2name
)
train_set = train_set + train_set_bal

# Create validation split from eval
# Since the code for BEATs evals is not public, it is hard to estimate how they create
# val set. However, it seems like AST is using all of the training data. To replicate this
# setup we do not use remove any data from train set and use 10% of eval set as val set.
# This results in ~1% gain in mAP score.
# AST- https://github.com/YuanGongND/ast/tree/master
random.seed(42)
random.shuffle(eval_set)
total_len = len(eval_set)
val_len = total_len // 10
val_set = eval_set[:val_len]

print(f"Train set size: {len(train_set)}")
print(f"Val set size: {len(val_set)}")
print(f"Eval set size: {len(eval_set)}")

data_sampling_weights = generate_sampling_weights(train_set, mid2name)

for dataset, name in [(train_set, "train"), (val_set, "val"), (eval_set, "eval")]:
    missing_wav_file = 0
    text_write_path = os.path.join(DATA_WRITE_ROOT, name, "text")
    wav_scp_write_path = os.path.join(DATA_WRITE_ROOT, name, "wav.scp")
    utt2spk_write_path = os.path.join(DATA_WRITE_ROOT, name, "utt2spk")

    os.makedirs(os.path.dirname(text_write_path), exist_ok=True)
    os.makedirs(os.path.dirname(wav_scp_write_path), exist_ok=True)
    os.makedirs(os.path.dirname(utt2spk_write_path), exist_ok=True)
    sample_weight_f = (
        open(os.path.join(DATA_WRITE_ROOT, "utt2weight"), "w")
        if name == "train"
        else None
    )

    with open(text_write_path, "w") as text_f, open(
        wav_scp_write_path, "w"
    ) as wav_f, open(utt2spk_write_path, "w") as utt2spk_f:
        for uttid, item in enumerate(tqdm(dataset, desc=f"Processing {name} set")):
            wav_directory = item["wav_directory"]
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
            if sample_weight_f:
                print(
                    f"as20k-{name}-{uttid} {data_sampling_weights[uttid]}",
                    file=sample_weight_f,
                )
    print(f"Missing {missing_wav_file} wav files in {name} set.")

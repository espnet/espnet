#!/usr/bin/env python3

# Copyright 2021 Carnegie Mellon University (Yifan Peng)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Speech Commands Dataset: https://arxiv.org/abs/1804.03209


import argparse
import os
import os.path

import numpy as np

parser = argparse.ArgumentParser(
    description="Process speech commands dataset with 35 commands."
)
parser.add_argument(
    "--data_path",
    type=str,
    default="downloads/speech_commands_v0.02",
    help="folder containing the original data",
)
parser.add_argument(
    "--train_dir",
    type=str,
    default="data/train",
    help="output folder for training data",
)
parser.add_argument(
    "--dev_dir", type=str, default="data/dev", help="output folder for validation data"
)
parser.add_argument(
    "--test_dir", type=str, default="data/test", help="output folder for test data"
)
args = parser.parse_args()


SAMPLE_RATE = 16000
WORDS = [
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "bed",
    "bird",
    "cat",
    "dog",
    "happy",
    "house",
    "marvin",
    "sheila",
    "tree",
    "wow",
    "backward",
    "forward",
    "follow",
    "learn",
    "visual",
]  # 35 commands


# Generate train and dev data
with open(os.path.join(args.data_path, "validation_list.txt"), "r") as dev_f:
    dev_file_list = [line.rstrip() for line in dev_f.readlines()]
    dev_file_list = [
        os.path.abspath(os.path.join(args.data_path, line)) for line in dev_file_list
    ]
with open(os.path.join(args.data_path, "testing_list.txt"), "r") as test_f:
    test_file_list = [line.rstrip() for line in test_f.readlines()]
    test_file_list = [
        os.path.abspath(os.path.join(args.data_path, line)) for line in test_file_list
    ]

full_file_list = []
for word in WORDS:
    for wav_file in os.listdir(os.path.join(args.data_path, word)):
        if wav_file.endswith(".wav"):
            full_file_list.append(
                os.path.abspath(os.path.join(args.data_path, word, wav_file))
            )

train_file_list = list(set(full_file_list) - set(dev_file_list) - set(test_file_list))
assert len(train_file_list) + len(dev_file_list) + len(test_file_list) == len(
    full_file_list
)

for name in ["train", "dev", "test"]:
    if name == "train":
        file_list = train_file_list
        out_dir = args.train_dir
    elif name == "dev":
        file_list = dev_file_list
        out_dir = args.dev_dir
    else:
        file_list = test_file_list
        out_dir = args.test_dir

    with open(os.path.join(out_dir, "text"), "w") as text_f, open(
        os.path.join(out_dir, "wav.scp"), "w"
    ) as wav_scp_f, open(os.path.join(out_dir, "utt2spk"), "w") as utt2spk_f:
        for wav_abspath in file_list:  # absolute path
            word, wav = wav_abspath.split("/")[-2:]
            uttid = f"{word}_{wav[:-4]}"
            text_f.write(uttid + " " + word + "\n")
            wav_scp_f.write(uttid + " " + wav_abspath + "\n")
            utt2spk_f.write(uttid + " " + uttid + "\n")

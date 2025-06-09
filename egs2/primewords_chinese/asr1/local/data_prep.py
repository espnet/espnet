#!/usr/bin/env python3

# Copyright 2021 Carnegie Mellon University (Yifan Peng)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import glob
import json
import math
import os
import os.path

parser = argparse.ArgumentParser(description="Prepare Primewords_Chinese")
parser.add_argument(
    "--data_path", type=str, help="Path to the directory containing all files"
)
parser.add_argument("--train_dir", type=str, help="Path to the train data")
parser.add_argument("--dev_dir", type=str, help="Path to the dev data")
parser.add_argument("--test_dir", type=str, help="Path to the test data")
parser.add_argument("--dev_ratio", type=float, default=0.136, help="Ratio of dev set")
parser.add_argument("--test_ratio", type=float, default=0.136, help="Ratio of test set")
args = parser.parse_args()


with open(os.path.join(args.data_path, "set1_transcript.json"), "r") as fp:
    transcript = json.load(fp)  # list
user_ids = sorted(list(set([e["user_id"] for e in transcript])))

n_dev_users = math.floor(args.dev_ratio * len(user_ids))
n_test_users = math.floor(args.test_ratio * len(user_ids))
dev_users = user_ids[:n_dev_users]
test_users = user_ids[n_dev_users : n_dev_users + n_test_users]
train_users = user_ids[n_dev_users + n_test_users :]
assert len(set(dev_users).intersection(set(test_users))) == 0
assert len(set(dev_users).intersection(set(train_users))) == 0
assert len(set(test_users).intersection(set(train_users))) == 0

wav_list = glob.glob(os.path.join(args.data_path, "audio_files/*/*/*.wav"))
wavname2abspath = {os.path.basename(e): os.path.abspath(e) for e in wav_list}

train_samples = []
dev_samples = []
test_samples = []
for sample in transcript:
    processed_sample = {
        "user_id": sample["user_id"],  # speaker id
        "text": "".join(sample["text"].split()),  # remove white spaces
        "id": sample["id"],  # utterance id
        "abs_path": wavname2abspath[sample["file"]],
    }
    if processed_sample["user_id"] in train_users:
        train_samples.append(processed_sample)
    elif processed_sample["user_id"] in dev_users:
        dev_samples.append(processed_sample)
    elif processed_sample["user_id"] in test_users:
        test_samples.append(processed_sample)
    else:
        raise RuntimeError

for setname in ["train", "dev", "test"]:
    if setname == "train":
        sample_list = train_samples
        dest_dir = args.train_dir
    elif setname == "dev":
        sample_list = dev_samples
        dest_dir = args.dev_dir
    elif setname == "test":
        sample_list = test_samples
        dest_dir = args.test_dir
    else:
        raise RuntimeError

    with open(os.path.join(dest_dir, "text"), "w") as text_f, open(
        os.path.join(dest_dir, "wav.scp"), "w"
    ) as wav_scp_f, open(os.path.join(dest_dir, "utt2spk"), "w") as utt2spk_f:
        for sample in sample_list:
            text_f.write(f"{int(sample['id']):08d} {sample['text']}\n")
            wav_scp_f.write(f"{int(sample['id']):08d} {sample['abs_path']}\n")
            utt2spk_f.write(f"{int(sample['id']):08d} {int(sample['id']):08d}\n")

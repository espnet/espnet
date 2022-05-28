#!/usr/bin/env bash

# Copyright 2021  Sujay Suresh Kumar
#           2021  Carnegie Mellon University
# Apache 2.0

import json
import os
import string as string_lib
import sys
from pathlib import Path

if len(sys.argv) != 2:
    print("Usage: python data_prep.py [catslu_root]")
    sys.exit(1)

catslu_root = sys.argv[1]

BLACKLIST_IDS = ["map-df61ee397d015314dfde80255365428b_4b3d3b2f332793052a000014-1"]


def word_segmentation(string):
    new_string = ""
    for char in list(string):
        if char.encode("UTF-8").isalpha():
            new_string += char
        elif char in string_lib.punctuation:
            new_string += " "
        elif char in ["·", "，"]:
            new_string += " "
        else:
            new_string += " " + char + " "
    result = " ".join(new_string.strip().split())
    return result


catslu_root_path = Path(catslu_root)

# Here, we are considering only the MAP dataset of CATSLU
catslu_traindev = Path(os.path.join(catslu_root_path, "catslu_traindev", "data"))
catslu_traindev_domain_dirs = [
    f for f in catslu_traindev.iterdir() if f.is_dir() and f.name == "map"
]

catslu_test = Path(os.path.join(catslu_root_path, "catslu_test", "data"))
catslu_test_domain_dirs = [
    f for f in catslu_test.iterdir() if f.is_dir() and f.name == "map"
]

train_text = []
dev_text = set()
test_text = set()

train_wav_scp = []
dev_wav_scp = set()
test_wav_scp = set()

train_utt2spk = []
dev_utt2spk = set()
test_utt2spk = set()

train_labels = set()
test_labels = set()
valid_labels = set()


def _process_data(data):
    global BLACKLIST_IDS
    text = []
    wav_scp = []
    utt2spk = []
    for dialogue in data:
        dialogue_id = dialogue["dlg_id"]
        for utterance in dialogue["utterances"]:
            wav_path = os.path.join(
                domain_dir, "audios/{}.wav".format(utterance["wav_id"])
            )
            utt_id = "{}-{}-{}".format(
                domain_dir.parts[-1], utterance["wav_id"], utterance["utt_id"]
            )
            slots = ["none", "none", "none"]
            try:
                slots[0] = (
                    utterance["semantic"][0][0]
                    if utterance["semantic"] and len(utterance["semantic"][0]) > 0
                    else "none"
                )
                slots[1] = (
                    utterance["semantic"][0][1]
                    if utterance["semantic"] and len(utterance["semantic"][0]) > 1
                    else "none"
                )
            except Exception as e:
                print(e)

            intent = "_".join(slots)
            transcription = (
                intent + " " + word_segmentation(utterance["manual_transcript"])
            )
            if utt_id in BLACKLIST_IDS:
                continue
            text.append("{} {}".format(utt_id, transcription))
            wav_scp.append("{} {}".format(utt_id, wav_path))
            utt2spk.append("{} {}".format(utt_id, dialogue_id))

    return text, wav_scp, utt2spk


train_text = []
train_wav_scp = []
train_utt2spk = []
for domain_dir in catslu_traindev_domain_dirs:
    with open(os.path.join(domain_dir, "train.json")) as fp:
        train_data = json.load(fp)

    train_text, train_wav_scp, train_utt2spk = _process_data(train_data)

    with open(os.path.join(domain_dir, "development.json")) as fp:
        dev_data = json.load(fp)

    dev_text, dev_wav_scp, dev_utt2spk = _process_data(dev_data)


for domain_dir in catslu_test_domain_dirs:
    with open(os.path.join(domain_dir, "test.json")) as fp:
        test_data = json.load(fp)

    test_text, test_wav_scp, test_utt2spk = _process_data(test_data)


# Write train data
with open(os.path.join("data", "train", "text"), "w") as fp:
    fp.truncate()
    for line in train_text:
        fp.write(line + "\n")

with open(os.path.join("data", "train", "wav.scp"), "w") as fp:
    fp.truncate()
    for line in train_wav_scp:
        fp.write(line + "\n")

with open(os.path.join("data", "train", "utt2spk"), "w") as fp:
    fp.truncate()
    for line in train_utt2spk:
        fp.write(line + "\n")


# Write valid data
with open(os.path.join("data", "valid", "text"), "w") as fp:
    fp.truncate()
    for line in dev_text:
        fp.write(line + "\n")

with open(os.path.join("data", "valid", "wav.scp"), "w") as fp:
    fp.truncate()
    for line in dev_wav_scp:
        fp.write(line + "\n")

with open(os.path.join("data", "valid", "utt2spk"), "w") as fp:
    fp.truncate()
    for line in dev_utt2spk:
        fp.write(line + "\n")


# Write test data
with open(os.path.join("data", "test", "text"), "w") as fp:
    fp.truncate()
    for line in test_text:
        fp.write(line + "\n")

with open(os.path.join("data", "test", "wav.scp"), "w") as fp:
    fp.truncate()
    for line in test_wav_scp:
        fp.write(line + "\n")

with open(os.path.join("data", "test", "utt2spk"), "w") as fp:
    fp.truncate()
    for line in test_utt2spk:
        fp.write(line + "\n")

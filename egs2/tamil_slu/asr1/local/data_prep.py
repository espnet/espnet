#!/usr/bin/env bash

# Copyright 2021  Sujay Suresh Kumar
#           2021  Carnegie Mellon University
# Apache 2.0

import csv
import json
import os
import random
import string as string_lib
import sys
from collections import Counter
from pathlib import Path

if len(sys.argv) != 2:
    print("Usage: python data_prep.py [tamil_root]")
    sys.exit(1)

tamil_root = sys.argv[1]

tamil_root_path = Path(tamil_root)

with open(os.path.join(tamil_root, "Tamil_Data.csv"), "r") as fp:
    data = csv.reader(fp)
    global_data = [i for i in data][1:]

random.shuffle(global_data)
train_corpus = global_data[: int(0.8 * len(global_data))]
dev_corpus = global_data[int(0.8 * len(global_data)) : int(0.9 * len(global_data))]
test_corpus = global_data[int(0.9 * len(global_data)) :]

with open(os.path.join(tamil_root, "Tamil_Sentences.csv"), "r") as fp:
    data = csv.reader(fp)
    transcription_data = [i for i in data][1:]

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
    text = []
    wav_scp = []
    utt2spk = []
    for row in data:
        wav_path = os.path.join(tamil_root, "audio_files", row[0])
        utt_id = row[0].replace("/", "-")
        spk_id = row[0].split("/")[0]

        transcription_row = None
        for i in transcription_data:
            if i[0] == row[1] and i[2] == row[2]:
                intent = i[1].replace(".", "").replace(" ", "_")
                sentence = i[3]
                break

        # Skipping transcription because of noise
        # transcription = intent + " " + sentence
        transcription = intent

        text.append("{} {}".format(utt_id, transcription))
        wav_scp.append("{} {}".format(utt_id, wav_path))
        utt2spk.append("{} {}".format(utt_id, spk_id))

    return text, wav_scp, utt2spk


train_text = []
train_wav_scp = []
train_utt2spk = []
train_text, train_wav_scp, train_utt2spk = _process_data(train_corpus)
assert len(train_text) == len(train_wav_scp) == len(train_utt2spk)
dev_text, dev_wav_scp, dev_utt2spk = _process_data(dev_corpus)
assert len(dev_text) == len(dev_wav_scp) == len(dev_utt2spk)
test_text, test_wav_scp, test_utt2spk = _process_data(test_corpus)
assert len(test_text) == len(test_wav_scp) == len(test_utt2spk)
print(len(train_text))
print(len(dev_text))
print(len(test_text))

# See distribution of class labels

train_intents = Counter([i.split()[1] for i in train_text])
train_dist = {}
for intent, count in train_intents.items():
    train_dist[intent] = count / sum(train_intents.values())

dev_intents = Counter([i.split()[1] for i in dev_text])
dev_dist = {}
for intent, count in dev_intents.items():
    dev_dist[intent] = count / sum(dev_intents.values())

test_intents = Counter([i.split()[1] for i in test_text])
test_dist = {}
for intent, count in test_intents.items():
    test_dist[intent] = count / sum(test_intents.values())

print("Train set distribution", train_dist)
print("Dev set distribution", dev_dist)
print("Test set distribution", test_dist)

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

#!/usr/bin/env python3

# Copyright 2021  Chaitanya Narisetty
#           2021  Carnegie Mellon University
# Apache 2.0


import os
import random
import sys
import librosa


if len(sys.argv) != 3:
    print("Usage: python prepare_data.py [data-directory] [language-ID]")
    sys.exit(1)

datadir = sys.argv[1]
lang = sys.argv[2]

traindir = f"{datadir}/{lang}-in-Train/"
testdir = f"{datadir}/{lang}-in-Test/"

train_datadir = f"data/train_{lang}/"
valid_datadir = f"data/dev_{lang}/"
test_datadir = f"data/test_{lang}/"

os.popen(f"mkdir -p {train_datadir}").read()
os.popen(f"mkdir -p {valid_datadir}").read()
os.popen(f"mkdir -p {test_datadir}").read()


# prepare data for training and validation splits
with open(traindir + "transcription.txt") as f:
    train_lines = [line.rstrip() for line in f.readlines()]
    train_id2text = {}
    train_id2filepath = {}
    for line in train_lines:
        wav_id = line.split()[0]
        filepath = f"{traindir}/Audios/{wav_id}.wav"
        train_id2text[wav_id] = " ".join(line.split()[1:])
        train_id2filepath[wav_id] = filepath

wav_ids = list(train_id2text.keys())
random.shuffle(wav_ids)
valid_id2text = {}
valid_totaldur = 2 * 60 * 60  # (in seconds) 2 hours taken for validation split
for wav_id in wav_ids:
    dur = librosa.get_duration(filename=train_id2filepath[wav_id])
    valid_id2text[wav_id] = train_id2text.pop(wav_id)
    valid_totaldur -= dur
    if valid_totaldur < 0:
        break


with open(train_datadir + "text", "w") as f:
    for wav_id in sorted(train_id2text):
        f.write(f"{lang}_{wav_id} {train_id2text[wav_id]}\n")
with open(train_datadir + "wav.scp", "w") as f:
    for wav_id in sorted(train_id2text):
        f.write(f"{lang}_{wav_id} {train_id2filepath[wav_id]}\n")
with open(train_datadir + "spk2utt", "w") as f:
    for wav_id in sorted(train_id2text):
        f.write(f"spk_{lang}_{wav_id} {lang}_{wav_id}\n")
with open(train_datadir + "utt2spk", "w") as f:
    for wav_id in sorted(train_id2text):
        f.write(f"{lang}_{wav_id} spk_{lang}_{wav_id}\n")

with open(valid_datadir + "text", "w") as f:
    for wav_id in sorted(valid_id2text):
        f.write(f"{lang}_{wav_id} {valid_id2text[wav_id]}\n")
with open(valid_datadir + "wav.scp", "w") as f:
    for wav_id in sorted(valid_id2text):
        f.write(f"{lang}_{wav_id} {train_id2filepath[wav_id]}\n")
with open(valid_datadir + "spk2utt", "w") as f:
    for wav_id in sorted(valid_id2text):
        f.write(f"spk_{lang}_{wav_id} {lang}_{wav_id}\n")
with open(valid_datadir + "utt2spk", "w") as f:
    for wav_id in sorted(valid_id2text):
        f.write(f"{lang}_{wav_id} spk_{lang}_{wav_id}\n")


# prepare test data
with open(testdir + "transcription.txt") as f:
    test_lines = [line.rstrip() for line in f.readlines()]
    test_id2text = {}
    test_id2filepath = {}
    for line in test_lines:
        wav_id = line.split()[0]
        filepath = f"{testdir}/Audios/{wav_id}.wav"
        test_id2text[wav_id] = " ".join(line.split()[1:])
        test_id2filepath[wav_id] = filepath

with open(test_datadir + "text", "w") as f:
    for wav_id in sorted(test_id2text):
        f.write(f"{lang}_{wav_id} {test_id2text[wav_id]}\n")
with open(test_datadir + "wav.scp", "w") as f:
    for wav_id in sorted(test_id2text):
        f.write(f"{lang}_{wav_id} {test_id2filepath[wav_id]}\n")
with open(test_datadir + "spk2utt", "w") as f:
    for wav_id in sorted(test_id2text):
        f.write(f"spk_{lang}_{wav_id} {lang}_{wav_id}\n")
with open(test_datadir + "utt2spk", "w") as f:
    for wav_id in sorted(test_id2text):
        f.write(f"{lang}_{wav_id} spk_{lang}_{wav_id}\n")

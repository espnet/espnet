#!/usr/bin/env bash

# Copyright 2023  Siddhant Arora
#           2023  Carnegie Mellon University
# Apache 2.0


import os
import random
import sys
from pathlib import Path

if len(sys.argv) != 2:
    print("Usage: python data_prep.py [voxforge_cmd_root]")
    sys.exit(1)
voxforge_cmd_root = sys.argv[1]
train_list, valid_list, test_list = [], [], []
for download in Path(voxforge_cmd_root, "extracted").iterdir():
    audios = []
    for speaker in Path(download).iterdir():
        entry = Path(speaker, "wav")
        new_audios = list(entry.glob("*.wav"))
        audios += new_audios
        if len(audios) > 1800:
            break
    random.seed(1)
    random.shuffle(audios)
    count = 0
    for audio_path in audios[:1200]:
        train_list.append((download.name, audio_path))
    for audio_path in audios[1200:1500]:
        valid_list.append((download.name, audio_path))
    for audio_path in audios[1500:1800]:
        test_list.append((download.name, audio_path))
dir_dict = {
    "train": train_list,
    "valid": valid_list,
    "test": test_list,
}

for x in dir_dict:
    with open(os.path.join("data", x, "wav.scp"), "w") as wav_scp_f, open(
        os.path.join("data", x, "utt2spk"), "w"
    ) as utt2spk_f, open(os.path.join("data", x, "text"), "w") as text_f:
        for line in dir_dict[x]:
            cls = line[0]
            utt_id = "_".join(str(line[1]).split("/")[-3:])
            spk = str(line[1]).split("/")[-3]

            wav_scp_f.write(utt_id + " " + str(line[1]) + "\n")
            text_f.write(utt_id + " " + cls + "\n")
            utt2spk_f.write(utt_id + " " + spk + "\n")

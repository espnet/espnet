#!/usr/bin/env bash

# Copyright 2023  Siddhant Arora
#           2023  Carnegie Mellon University
# Apache 2.0

import json
import os
import pickle
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split

if len(sys.argv) != 2:
    print("Usage: python data_prep.py [ASVSpoof_root]")
    sys.exit(1)
esc_root = sys.argv[1]

meta_data = pd.read_csv(Path(esc_root, "meta", "esc50.csv"))
split_df = {}
split_df["test"] = meta_data[meta_data["fold"] == 1]
train_val_df = meta_data[meta_data["fold"] != 1]
split_df["train"], split_df["valid"] = train_test_split(
    train_val_df, test_size=0.2, random_state=1
)
dir_dict = split_df
for x in dir_dict:
    with open(os.path.join("data", x, "wav.scp"), "w") as wav_scp_f, open(
        os.path.join("data", x, "utt2spk"), "w"
    ) as utt2spk_f, open(os.path.join("data", x, "text"), "w") as text_f:
        filename = dir_dict[x]["filename"].values.tolist()
        label = dir_dict[x]["target"].values.tolist()
        for line_count in range(len(filename)):
            cls = "audio_class:" + str(label[line_count])
            utt_id = filename[line_count].replace(".wav", "")
            spk = utt_id
            data_dir = Path(esc_root, "audio", filename[line_count])

            wav_scp_f.write(utt_id + " " + str(data_dir) + "\n")
            text_f.write(utt_id + " " + cls + "\n")
            utt2spk_f.write(utt_id + " " + spk + "\n")

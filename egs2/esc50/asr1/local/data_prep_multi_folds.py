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
from torch.utils.data import random_split

if len(sys.argv) < 2:
    print(len(sys.argv))
    print(
        "Usage: python data_prep.py [ESC-50_root] [FOLD] [ROOT] where "
        "FOLD and ROOT are optional"
    )
    sys.exit(1)

esc_root = sys.argv[1]
fold_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1
data_prep_root = sys.argv[3] if len(sys.argv) == 4 else "."

meta_data = pd.read_csv(Path(esc_root, "meta", "esc50.csv"))

split_df = {}
split_df[f"val{fold_num}"] = meta_data[meta_data["fold"] == fold_num]
split_df[f"train{fold_num}"] = meta_data[meta_data["fold"] != fold_num]

print(
    "For fold number:",
    fold_num,
    "Train and Val split",
    len(split_df[f"train{fold_num}"]),
    len(split_df[f"val{fold_num}"]),
)

dir_dict = split_df
for x in dir_dict:
    os.makedirs(os.path.join(data_prep_root, "data", x), exist_ok=True)
    with open(
        os.path.join(data_prep_root, "data", x, "wav.scp"), "w"
    ) as wav_scp_f, open(
        os.path.join(data_prep_root, "data", x, "utt2spk"), "w"
    ) as utt2spk_f, open(
        os.path.join(data_prep_root, "data", x, "text"), "w"
    ) as text_f:
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

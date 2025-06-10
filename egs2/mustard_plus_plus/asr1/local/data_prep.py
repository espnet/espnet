#!/usr/bin/env bash

# Copyright 2023  Siddhant Arora
#           2023  Carnegie Mellon University
# Apache 2.0

import json
import os
import pickle
import sys
from pathlib import Path

import nlp2
import pandas as pd
from torch.utils.data import random_split

if len(sys.argv) != 2:
    print("Usage: python data_prep.py [mustard_plus_plus_root]")
    sys.exit(1)
mustard_plus_plus_root = sys.argv[1]
data = pd.read_csv(mustard_plus_plus_root + "/mustard++_text.csv", index_col="KEY")


files = list(nlp2.get_files_from_dir(mustard_plus_plus_root, match="wav"))
val_test_len = int(len(files) / 10)
train_len = len(files) - 2 * val_test_len
dataset_split = random_split(files, [train_len, val_test_len, val_test_len])
train_fnames = dataset_split[0]
val_fnames = dataset_split[1]
test_fnames = dataset_split[2]
dir_dict = {
    "train": train_fnames,
    "valid": val_fnames,
    "test": test_fnames,
}
for x in dir_dict:
    with open(os.path.join("data", x, "wav.scp"), "w") as wav_scp_f, open(
        os.path.join("data", x, "utt2spk"), "w"
    ) as utt2spk_f, open(os.path.join("data", x, "text"), "w") as text_f:
        for line in dir_dict[x]:
            utt_id = line.split("/")[-1].replace(".wav", "")
            if data.loc[utt_id, "Sarcasm"]:
                cls = "class:sarcasm"
            else:
                cls = "class:not_sarcasm"
            spk = (
                data.loc[utt_id, "SHOW"]
                + "_"
                + data.loc[utt_id, "SPEAKER"]
                + "_speaker"
            )
            utt_id = spk + "_" + utt_id

            wav_scp_f.write(utt_id + " " + line + "\n")
            text_f.write(utt_id + " " + cls + "\n")
            utt2spk_f.write(utt_id + " " + spk + "\n")

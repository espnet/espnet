#!/usr/bin/env bash

# Copyright 2021  Siddhant Arora
#           2021  Carnegie Mellon University
# Apache 2.0


import os
import re
import sys

import pandas as pd

if len(sys.argv) != 2:
    print("Usage: python data_prep.py [data_root]")
    sys.exit(1)
data_root = sys.argv[1]

dir_dict = {
    "train": "fold1_train.csv",
    "test": "fold1_evaluate.csv",
}

for x in dir_dict:
    with open(os.path.join("data", x, "text"), "w") as text_f, open(
        os.path.join("data", x, "wav.scp"), "w"
    ) as wav_scp_f, open(os.path.join("data", x, "utt2spk"), "w") as utt2spk_f:

        text_f.truncate()
        wav_scp_f.truncate()
        utt2spk_f.truncate()
        transcript_df = pd.read_csv(
            os.path.join(data_root, "evaluation_setup", dir_dict[x]), sep="\t"
        )
        # lines = sorted(transcript_df.values, key=lambda s: s[0])
        for row in transcript_df.values:
            # print(row)
            words = row[1]
            # print(words)
            path = os.path.join(data_root, row[0])
            utt_id = row[0].split("/")[-1]
            # print(utt_id + " " + words + "\n")
            text_f.write(utt_id + " " + words + "\n")
            wav_scp_f.write(utt_id + " " + path + "\n")
            utt2spk_f.write(utt_id + " None \n")

#!/usr/bin/env bash

# Copyright 2021  Siddhant Arora
#           2021  Carnegie Mellon University
# Apache 2.0


import os
import re
import sys

import pandas as pd

if len(sys.argv) != 2:
    print("Usage: python data_prep.py [fsc_root]")
    sys.exit(1)
fsc_root = sys.argv[1]

dir_dict = {
    "train": "train_data.csv",
    "valid": "valid_data.csv",
    "test": "test_data.csv",
}

for x in dir_dict:
    with open(os.path.join("data", x, "text"), "w") as text_f, open(
        os.path.join("data", x, "wav.scp"), "w"
    ) as wav_scp_f, open(
        os.path.join("data", x, "transcript"), "w"
    ) as transcript_f, open(
        os.path.join("data", x, "utt2spk"), "w"
    ) as utt2spk_f:
        text_f.truncate()
        wav_scp_f.truncate()
        utt2spk_f.truncate()
        transcript_df = pd.read_csv(os.path.join(fsc_root, "data", dir_dict[x]))
        # lines = sorted(transcript_df.values, key=lambda s: s[0])
        for row in transcript_df.values:
            words = (
                row[4].replace(" ", "_")
                + "_"
                + row[5].replace(" ", "_")
                + "_"
                + row[6].replace(" ", "_")
                + " "
                + row[3].encode("ascii", "ignore").decode()
            )
            print(words)
            path_arr = row[1].split("/")
            utt_id = path_arr[-2] + "_" + path_arr[-1]
            # print(utt_id + " " + words + "\n")
            text_f.write(utt_id + " " + words + "\n")
            wav_scp_f.write(utt_id + " " + fsc_root + "/" + row[1] + "\n")
            utt2spk_f.write(utt_id + " " + row[2] + "\n")

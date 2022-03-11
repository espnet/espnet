#!/usr/bin/env bash

# Copyright 2021  Siddhant Arora
#           2021  Carnegie Mellon University
# Apache 2.0


import os
import re
import sys
import pandas as pd

if len(sys.argv) != 2:
    print("Usage: python data_prep.py [dstc2_root]")
    sys.exit(1)
dstc2_root = sys.argv[1]

dir_dict = {
    "train": "train.csv",
    "valid": "validation.csv",
    "test": "test.csv",
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
        transcript_df = pd.read_csv(os.path.join(dstc2_root, "data", dir_dict[x]))
        for row in transcript_df.values:
            words = (
                " <sep> ".join(sorted(eval(row[5])))
                + " <utt> "
                + row[2].replace("<unk>", "unk").encode("ascii", "ignore").decode()
            )
            path_arr = row[4].split("/")
            utt_id = path_arr[-2] + "_" + path_arr[-1]
            speaker_id = path_arr[-2]
            text_f.write(utt_id + " " + words + "\n")
            wav_scp_f.write(utt_id + " " + dstc2_root + "/" + row[4] + "\n")
            utt2spk_f.write(utt_id + " " + speaker_id + "\n")

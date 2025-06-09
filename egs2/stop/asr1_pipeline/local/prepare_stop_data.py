#!/usr/bin/env bash

# Copyright 2021  Siddhant Arora
#           2021  Carnegie Mellon University
# Apache 2.0


import os
import re
import sys

import pandas as pd

if len(sys.argv) != 2:
    print("Usage: python data_prep.py [stop_root]")
    sys.exit(1)
stop_root = sys.argv[1]

dir_dict = {
    "train": "manifests/train.tsv",
    "valid": "manifests/eval.tsv",
    "test": "manifests/test.tsv",
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
        transcript_df = pd.read_csv(os.path.join(stop_root, dir_dict[x]), sep="\t")
        # lines = sorted(transcript_df.values, key=lambda s: s[0])
        for row in transcript_df.values:
            if str(row[-1]) == "nan":
                continue
            words = row[-3].lower()
            # print(words)
            path_arr = row[0].split("/")
            utt_id = path_arr[-2] + "_" + path_arr[-1]
            # print(utt_id + " " + words + "\n")
            text_f.write(utt_id + " " + words + "\n")
            wav_scp_f.write(
                utt_id
                + " "
                + stop_root
                + "/"
                + row[0].replace("_eval_0", "_eval").replace("_test_0", "_test")
                + "\n"
            )
            utt2spk_f.write(utt_id + " " + utt_id + "_1 \n")

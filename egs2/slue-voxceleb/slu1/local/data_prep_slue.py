#!/usr/bin/env bash

# Copyright 2021  Siddhant Arora
#           2021  Carnegie Mellon University
# Apache 2.0


import os
import re
import sys

import pandas as pd

if len(sys.argv) != 4:
    print("Usage: python data_prep.py [root]")
    sys.exit(1)
root = sys.argv[1]
if sys.argv[2].lower() == "false":
    use_classifier = False
else:
    use_classifier = True
if sys.argv[3].lower() == "false":
    run_only_asr = False
else:
    run_only_asr = True

dir_dict = {
    "train": "fine-tune.tsv",
    "devel": "dev.tsv",
    "test": "test.tsv",
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
        transcript_df = pd.read_csv(os.path.join(root, dir_dict[x]), sep="\t")
        for row in transcript_df.values:
            if row[4] == "<mixed>":
                continue
            if row[4] == "Disagreement":
                continue
            if use_classifier:
                words = row[4].replace(" ", "_")
            elif run_only_asr:
                words = row[1].encode("ascii", "ignore").decode()
            else:
                words = (
                    row[4].replace(" ", "_")
                    + " "
                    + row[1].encode("ascii", "ignore").decode()
                )
            speaker = row[2]
            if x == "train":
                path = "audio/fine-tune_raw/" + row[0] + ".flac"
            elif x == "devel":
                path = "audio/dev_raw/" + row[0] + ".flac"
            else:
                path = "audio/test_raw/" + row[0] + ".flac"
            utt_id = row[0]
            text_f.write(utt_id + " " + words + "\n")
            wav_scp_f.write(utt_id + " " + root + "/" + path + "\n")
            utt2spk_f.write(utt_id + " " + speaker + "\n")

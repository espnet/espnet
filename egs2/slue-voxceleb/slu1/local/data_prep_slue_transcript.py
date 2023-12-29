#!/usr/bin/env bash

# Copyright 2021  Siddhant Arora
#           2021  Carnegie Mellon University
# Apache 2.0


import os
import re
import sys

import pandas as pd

if len(sys.argv) != 3:
    print("Usage: python data_prep.py [root] [transcript_folder]")
    sys.exit(1)
root = sys.argv[1]
transcript_folder = sys.argv[2]

dir_dict = {
    "train": "slue-voxceleb_fine-tune.tsv",
    "devel": "slue-voxceleb_dev.tsv",
    "test": "slue-voxceleb_test_blind.tsv",
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
        asr_transcript_file = open(os.path.join(transcript_folder, x, "text"))
        asr_transcript_dict = {}
        for line in asr_transcript_file:
            if len(line.split()) == 2:
                text = "<blank>"
            else:
                text = line.split()[2].replace("▁", "")
                for sub_word in line.split()[3:]:
                    if "▁" in sub_word:
                        text = text + " " + sub_word.replace("▁", "")
                    else:
                        text = text + sub_word
            asr_transcript_dict[line.split()[0]] = text
        for row in transcript_df.values:
            if x == "test":
                speaker = row[1]
                words = (
                    "<blank>"  # Test set is blind, will have to submit to leaderboard
                )
            else:
                if row[4] == "<mixed>":
                    continue
                words = (
                    row[4].replace(" ", "_")
                    + " "
                    + row[1].encode("ascii", "ignore").decode()
                )
                speaker = row[2]
            if x == "train":
                path = "fine-tune_raw/" + row[0] + ".flac"
            elif x == "devel":
                path = "dev_raw/" + row[0] + ".flac"
            else:
                path = "test_raw/" + row[0] + ".flac"
            utt_id = row[0]
            text_f.write(utt_id + " " + words + "\n")
            wav_scp_f.write(utt_id + " " + root + "/" + path + "\n")
            utt2spk_f.write(utt_id + " " + speaker + "\n")
            transcript_f.write(utt_id + " " + asr_transcript_dict[utt_id] + "\n")

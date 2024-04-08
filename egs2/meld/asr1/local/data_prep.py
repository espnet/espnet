#!/usr/bin/env bash

# Copyright 2022  Ziang Zhou
#           2022  Carnegie Mellon University
# Apache 2.0

import os
import re
import sys

import pandas as pd

if len(sys.argv) != 2:
    print("Usage: python data_prep.py [meld_root]")
    sys.exit(1)

meld_root = sys.argv[1]
phases = ["train", "valid", "test"]

for phase in phases:
    os.makedirs(os.path.join("data", phase), exist_ok=True)
    phase_df = pd.read_csv(os.path.join(meld_root, "data", f"{phase}_sent_emo.csv"))
    phase_len = len(phase_df)
    utt2spk = []
    text = []
    wav_scp = []

    for i in range(phase_len):
        line = list(phase_df.iloc[i])
        # line = sorted_df[i]
        spk = (
            line[2]
            .replace(". ", ".")
            .replace(".", "_")
            .replace(" ", "_")
            .replace("\x92", "'")
            .replace("-", "_")
            .replace("'", "_")
            .replace("/", "_")
        )
        # filter out extremely long sequence
        if f"{spk}-dia{line[5]}-utt{line[6]}-sea{line[7]}-epi{line[8]}-{phase}" in [
            "Ross-dia125-utt3-sea4-epi18-train",
            "Phoebe-dia110-utt7-sea6-epi11-valid",
            "Student-dia38-utt4-sea3-epi7-test",
            "Phoebe-dia220-utt0-sea2-epi1-test",
        ]:
            continue

        curr_line = " ".join(
            re.sub(r"([.,!?()\-\"])", r" ", line[1])
            .encode("ascii", "ignore")
            .decode()
            .lower()
            .split()
        )
        utt2spk.append(
            f"{spk}-dia{line[5]}-utt{line[6]}-sea{line[7]}-epi{line[8]}-{phase} {spk}"
        )
        text.append(
            f"{spk}-dia{line[5]}-utt{line[6]}-sea{line[7]}-epi{line[8]}-{phase} "
            f"{line[3]} {curr_line}"
        )
        wav_scp.append(
            f"{spk}-dia{line[5]}-utt{line[6]}-sea{line[7]}-epi{line[8]}-{phase} "
            f"ffmpeg -i {meld_root}/wavs/{phase}/dia{line[5]}_utt{line[6]}.mp4 -ac 1 "
            f"-ar 16000 -f wav -vn -hide_banner -loglevel error - |"
        )

    with open(os.path.join("data", phase, "utt2spk"), "w") as f:
        f.write("\n".join(utt2spk))
    with open(os.path.join("data", phase, "text"), "w") as f:
        f.write("\n".join(text))
    with open(os.path.join("data", phase, "wav.scp"), "w") as f:
        f.write("\n".join(wav_scp))

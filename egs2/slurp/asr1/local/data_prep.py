#!/usr/bin/env bash

# Copyright 2021  Siddhant Arora
#           2021  Carnegie Mellon University
# Apache 2.0


import os
import re
import sys
import json
import subprocess
import pandas as pd

if len(sys.argv) != 2:
    print("Usage: python data_prep.py [slurp_root]")
    sys.exit(1)
slurp_root = sys.argv[1]

dir_dict = {
    "train": "train.jsonl",
    "valid": "dev.jsonl",
    "test": "test.jsonl",
}

spk = {}

with open(os.path.join(slurp_root, "dataset", "slurp", "metadata" + ".json")) as meta:
    records = json.load(meta)
    for record in records.values():
        for filename in record["recordings"].keys():
            spk[filename[6:-5]] = record["recordings"][filename]["usrid"]
            print(spk)
            exit()

for x in dir_dict:
    with open(os.path.join("data", x, "text"), "w") as text_f, open(
        os.path.join("data", x, "wav.scp"), "w"
    ) as wav_scp_f, open(os.path.join("data", x, "utt2spk"), "w") as utt2spk_f:

        text_f.truncate()
        wav_scp_f.truncate()
        utt2spk_f.truncate()
        transcript_df = pd.read_csv(os.path.join(slurp_root, "data", dir_dict[x]))
        # lines = sorted(transcript_df.values, key=lambda s: s[0])
        for row in transcript_df.values:
            words = (
                row[4].replace(" ", "_")
                + "_"
                + row[5].replace(" ", "_")
                + "_"
                + row[6].replace(" ", "_")
            )
            path_arr = row[1].split("/")
            utt_id = path_arr[-2] + "_" + path_arr[-1]
            # print(utt_id + " " + words + "\n")
            text_f.write(utt_id + " " + words + "\n")
            wav_scp_f.write(utt_id + " " + fsc_root + "/" + row[1] + "\n")
            utt2spk_f.write(utt_id + " " + row[2] + "\n")

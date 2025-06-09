#!/usr/bin/env bash

# Copyright 2023  Siddhant Arora
#           2023  Carnegie Mellon University
# Apache 2.0


import os
import sys

if len(sys.argv) != 2:
    print("Usage: python data_prep.py [ar_sc]")
    sys.exit(1)
ar_sc_root = sys.argv[1]

dir_dict = {
    "train": "train_full.csv",
    "valid": "dev_full.csv",
    "test": "test_full.csv",
}

for x in dir_dict:
    with open(os.path.join("data", x, "wav.scp"), "w") as wav_scp_f, open(
        os.path.join("data", x, "text"), "w"
    ) as text_f, open(
        os.path.join(ar_sc_root, "SpeechAdvReprogram/Datasets/AR-SCR", dir_dict[x]), "r"
    ) as f_meta, open(
        os.path.join("data", x, "words"), "w"
    ) as f_words:
        cls_set = set()

        lines = f_meta.readlines()
        print(len(lines))
        for line in lines[1:]:
            utt_id, cls = line.strip().split("\t")
            utt_id = "/".join(utt_id.split("/")[-2:])
            data_dir = os.path.join(ar_sc_root, "AR_ SpeechCommands_Database", utt_id)
            # utt_id = cls + "_" + utt_id

            wav_scp_f.write(utt_id + " " + data_dir + "\n")
            text_f.write(utt_id + " " + cls + "\n")
            if cls not in cls_set:
                cls_set.add(cls)
        cls_set = sorted(list(cls_set))
        for cls in cls_set:
            f_words.write(cls + "\n")

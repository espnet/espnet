#!/usr/bin/env bash

# Copyright 2023  Siddhant Arora
#           2023  Carnegie Mellon University
# Apache 2.0


import os
import sys

if len(sys.argv) != 2:
    print("Usage: python data_prep.py [lt_speech_cmd_root]")
    sys.exit(1)
lt_speech_cmd_root = sys.argv[1]

dir_dict = {
    "train": "train_limit20.csv",
    "valid": "dev_full.csv",
    "test": "test_full.csv",
}

for x in dir_dict:
    with open(os.path.join("data", x, "wav.scp"), "w") as wav_scp_f, open(
        os.path.join("data", x, "utt2spk"), "w"
    ) as utt2spk_f, open(os.path.join("data", x, "text"), "w") as text_f, open(
        os.path.join(
            lt_speech_cmd_root, "SpeechAdvReprogram/Datasets/LT-SCR", dir_dict[x]
        ),
        "r",
    ) as f_meta:
        lines = f_meta.readlines()
        print(len(lines))
        for line in lines[1:]:
            utt_id, cls = line.strip().split("\t")
            spk = utt_id.split("/")[-1]
            utt_id = "/".join(utt_id.split("/")[-2:])
            data_dir = os.path.join(lt_speech_cmd_root, "dataset", utt_id)
            utt_id = spk + "_" + cls + "_" + utt_id

            wav_scp_f.write(utt_id + " " + data_dir + "\n")
            text_f.write(utt_id + " " + cls + "\n")
            utt2spk_f.write(utt_id + " " + spk + "\n")

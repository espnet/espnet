#!/usr/bin/env bash

# Copyright 2023  Siddhant Arora
#           2023  Carnegie Mellon University
# Apache 2.0


import os
import sys
from pathlib import Path

if len(sys.argv) != 2:
    print("Usage: python data_prep.py [ASVSpoof_root]")
    sys.exit(1)
ASVSpoof_root = sys.argv[1]

dir_dict = {
    "train": "ASVspoof2019.LA.cm.train.trn.txt",
}
mapping = {"train": "train"}
for x in dir_dict:
    with open(os.path.join("data", x, "wav.scp"), "w") as wav_scp_f, open(
        os.path.join("data", x, "utt2spk"), "w"
    ) as utt2spk_f, open(
        os.path.join(ASVSpoof_root, "LA/ASVspoof2019_LA_cm_protocols/", dir_dict[x]),
        "r",
    ) as f_meta:
        lines = f_meta.readlines()
        print(len(lines))
        split_path = Path(ASVSpoof_root, f"LA/ASVspoof2019_LA_{mapping[x]}", "flac")
        for line in lines:
            cls = line.split(" ")[4].replace("\n", "")
            spk = line.split(" ")[0]
            utt_id = spk + "_" + line.split(" ")[1]
            data_dir = Path(split_path, f'{line.split(" ")[1]}.flac')

            wav_scp_f.write(utt_id + " " + str(data_dir) + "\n")
            # text_f.write(utt_id + " " + cls + "\n")
            utt2spk_f.write(utt_id + " " + spk + "\n")

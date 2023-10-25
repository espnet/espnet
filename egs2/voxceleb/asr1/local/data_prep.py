#!/usr/bin/env bash

# Copyright 2023  Siddhant Arora
#           2023  Carnegie Mellon University
# Apache 2.0

import json
import os
import pickle
import sys
from pathlib import Path

from torch.utils.data import random_split

if len(sys.argv) != 2:
    print("Usage: python data_prep.py [voxceleb_root]")
    sys.exit(1)
voxceleb_root = sys.argv[1]
file1 = open(voxceleb_root + "/vox1_meta.csv")
line_arr = [line for line in file1]
gender_dict = {}
for line in line_arr[1:]:
    gender_dict[line.split("\t")[0]] = "gender:" + line.split("\t")[2]

dir_dict = {
    "train": "dev",
    "test": "test",
}
# mapping = {"train": "train", "valid": "dev", "test": "eval"}
for x in dir_dict:
    with open(os.path.join("data", x, "wav.scp"), "w") as wav_scp_f, open(
        os.path.join("data", x, "utt2spk"), "w"
    ) as utt2spk_f, open(os.path.join("data", x, "text"), "w") as text_f:
        dir_list = os.listdir(voxceleb_root + dir_dict[x])
        for dir_name in dir_list:
            if dir_name == "txt":
                continue
            file_list = os.listdir(voxceleb_root + dir_dict[x] + "/" + dir_name)
            for dir1 in file_list:
                file_list1 = os.listdir(
                    voxceleb_root + dir_dict[x] + "/" + dir_name + "/" + dir1
                )
                for dir2 in file_list1:
                    data_dir = (
                        voxceleb_root
                        + dir_dict[x]
                        + "/"
                        + dir_name
                        + "/"
                        + dir1
                        + "/"
                        + dir2
                    )
                    cls = gender_dict[dir_name]
                    spk = dir_name
                    utt_id = spk + "_" + dir1 + "_" + dir2

                    wav_scp_f.write(utt_id + " " + str(data_dir) + "\n")
                    text_f.write(utt_id + " " + cls + "\n")
                    utt2spk_f.write(utt_id + " " + spk + "\n")

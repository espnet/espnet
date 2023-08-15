#!/usr/bin/env bash

# Copyright 2023  Jee-weon Jung
#           2023  Carnegie Mellon University
# Apache 2.0

import os
import sys
from pathlib import Path
import pickle
import json
from torch.utils.data import random_split

if len(sys.argv) != 2:
    print("Usage: python data_prep.py [ASVSpoof_root]")
    sys.exit(1)
mustard_root = sys.argv[1]
split_file = os.path.join("/scratch/bbjs/shared/corpora/SpeechPrompt-v2/GSLM/preprocess/SD_mustard/", "split_indices.p")
with open(split_file, mode='rb') as file:
    split = pickle.load(file, encoding="latin1")
split_no = 0
train_val_idx = split[split_no][0]
test_idx = split[split_no][1]
with open(mustard_root+"/data/sarcasm_data.json", "r") as read_file:
    data = json.load(read_file)
fnames = list(data.keys())
train_val_fnames = [fnames[_id] for _id in train_val_idx]
test_fnames = [fnames[_id] for _id in test_idx]
val_len = int(len(train_val_fnames) / 10)
train_len = len(train_val_fnames) - val_len
train_fnames, val_fnames = random_split(train_val_fnames, [train_len, val_len])
train_fnames = list(train_fnames)
val_fnames = list(val_fnames)
# print(test_fnames[:10])
# print(train_fnames[:10])
# print(val_fnames[:10])
# exit()
dir_dict = {
    "train": train_fnames,
    "valid": val_fnames,
    "test": test_fnames,
}
# mapping = {"train": "train", "valid": "dev", "test": "eval"}
for x in dir_dict:
    with open(
        os.path.join("data", x, "wav.scp"), "w"
    ) as wav_scp_f, open(
        os.path.join("data", x, "utt2spk"), "w"
    ) as utt2spk_f, open(
        os.path.join("data", x, "text"), "w"
    ) as text_f:
        for line in dir_dict[x]:
            if data[line]['sarcasm']:
                cls='class:sarcasm'
            else:
                cls='class:not_sarcasm'
            spk=data[line]['show']+"_"+data[line]['speaker']+"_speaker"
            utt_id=spk+"_"+line
            # utt_id, cls = line.strip().split("\t")
            # spk=utt_id.split("/")[-1]
            # utt_id = "/".join(utt_id.split("/")[-2:])
            data_dir =  "/scratch/bbjs/shared/corpora/Mustard/utterances_final_wav/"+line+".wav"
            # utt_id = spk+"_"+ cls + "_" + utt_id

            wav_scp_f.write(utt_id + " " + str(data_dir) + "\n")
            text_f.write(utt_id + " " + cls + "\n")
            utt2spk_f.write(utt_id + " " +spk + "\n")

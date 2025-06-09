#!/usr/bin/env bash

# Copyright 2023  Siddhant Arora
#           2023  Carnegie Mellon University
# Apache 2.0


import os
import sys

from sklearn.model_selection import train_test_split

seed = 1337
os.environ["PYTHONHASHSEED"] = str(seed)


if len(sys.argv) != 2:
    print("Usage: python data_prep.py [acc_db]")
    sys.exit(1)
acc_db_root = sys.argv[1]

list_utts = []
for r, ds, fs in os.walk(acc_db_root):
    for f in fs:
        if not f.endswith(".wav"):
            continue
        f_dir = os.path.join(r, f)
        list_utts.append(f_dir)
print("total utt: ", len(list_utts))

list_train, list_test = train_test_split(
    list_utts, test_size=0.2, shuffle=True, random_state=seed
)
list_train, list_val = train_test_split(
    list_train, test_size=0.2, shuffle=True, random_state=seed
)
print("train, val, test", len(list_train), len(list_val), len(list_test))

dir_dict = {
    "train": list_train,
    "valid": list_val,
    "test": list_test,
}


for x in dir_dict:
    with open(os.path.join("data", x, "wav.scp"), "w") as wav_scp_f, open(
        os.path.join("data", x, "utt2spk"), "w"
    ) as f_utt2spk, open(os.path.join("data", x, "text"), "w") as text_f, open(
        os.path.join("data", x, "words"), "w"
    ) as f_words:
        cls_set = set()

        lines = dir_dict[x]
        for line in lines:
            utt_id = os.path.basename(line)
            speaker = "_".join(utt_id.split("_")[:2])
            cls = utt_id.split("_")[0]

            f_utt2spk.write(utt_id + " " + speaker + "\n")
            wav_scp_f.write(utt_id + " " + line + "\n")
            text_f.write(utt_id + " " + cls + "\n")

            if cls not in cls_set:
                cls_set.add(cls)
        cls_set = sorted(list(cls_set))
        for cls in cls_set:
            f_words.write(cls + "\n")

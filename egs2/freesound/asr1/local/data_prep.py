#!/usr/bin/env bash

# Copyright 2023  Siddhant Arora
#           2023  Carnegie Mellon University
# Apache 2.0

import json
import os
import pickle
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split

if len(sys.argv) != 2:
    print("Usage: python data_prep.py [ASVSpoof_root]")
    sys.exit(1)
esc_root = sys.argv[1]
dir_dict = {}
mapping = {"train": "training", "valid": "validation", "test": "testing"}
for split in ["train", "valid", "test"]:
    # Read manifest file generated from NeMo preprocessing script
    speech_data, background_data = [], []
    with open(
        Path(esc_root, f"manifest/balanced_speech_{mapping[split]}_manifest.json"), "r"
    ) as f:
        for line in f:
            speech_data.append(json.loads(line))
    with open(
        Path(esc_root, f"manifest/balanced_background_{mapping[split]}_manifest.json"),
        "r",
    ) as f:
        for line in f:
            background_data.append(json.loads(line))
    dir_dict[split] = speech_data + background_data

utt_id_dict = {}
for x in dir_dict:
    with open(os.path.join("data", x, "wav.scp"), "w") as wav_scp_f, open(
        os.path.join("data", x, "utt2spk"), "w"
    ) as utt2spk_f, open(os.path.join("data", x, "text"), "w") as text_f:
        for line in dir_dict[x]:
            wav_path = line["audio_filepath"]
            offset = line["offset"]
            duration = line["duration"]
            cls = "vad_class:" + line["label"]
            utt_id = (
                wav_path.split("/")[-2]
                + "_"
                + wav_path.split("/")[-1]
                + "_"
                + str(offset)
                + "_"
                + str(duration)
            )
            if utt_id not in utt_id_dict:
                utt_id_dict[utt_id] = line
            else:
                assert line == utt_id_dict[utt_id]
                count = 1
                utt_id = utt_id + "_" + str(count)
                while True:
                    if utt_id in utt_id_dict:
                        count += 1
                        utt_id = utt_id + "_" + str(count)
                    else:
                        utt_id_dict[utt_id] = line
                        break
            wav_path1 = str(wav_path).replace('"', '\\"')
            wav = (
                f' "{wav_path1}" -r 16k -t wav -'
                + f" trim {str(offset)} {str(duration)} | "
            )

            wav_scp_f.write(utt_id + "  sox " + wav + "\n")
            text_f.write(utt_id + " " + cls + "\n")
            utt2spk_f.write(utt_id + " " + utt_id + "\n")

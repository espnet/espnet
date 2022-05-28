#!/usr/bin/env bash

# Copyright 2021  Karthik Ganesan
#           2021  Carnegie Mellon University
# Apache 2.0


import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

if len(sys.argv) != 2:
    print("Usage: python data_prep.py [SINHALA]")
    sys.exit(1)
sinhala_root = sys.argv[1]


def read_sinhala_data(audio_csv, sentences_csv, export_csv):
    sent_df = pd.read_csv(sentences_csv)
    data = pd.read_csv(audio_csv)
    output_df = []

    if not os.path.exists("wavs"):
        os.mkdir("wavs")

    for i in range(len(sent_df)):
        intent, intent_details, inflection, transcript = (
            sent_df.iloc[i]["intent"],
            sent_df.iloc[i]["intent_details"],
            sent_df.iloc[i]["inflection"],
            sent_df.iloc[i]["sentence"],
        )

        for j in range(len(data)):
            wav_name, intent_, inflection_ = (
                data.iloc[j]["audio_file"],
                data.iloc[j]["intent"],
                data.iloc[j]["inflection"],
            )
            if intent_ == intent and inflection_ == inflection:
                # clean transcript
                # export audio of for the crop with wav_path_start_duration
                export_path = os.path.join("wavs", wav_name)
                # Append to output_df
                output_df.append(
                    [
                        export_path,
                        "unknown",
                        transcript,
                        intent_details.replace(" ", ""),
                    ]
                )

    X = pd.DataFrame(
        output_df, columns=["path", "speakerId", "transcription", "task_type"]
    )
    Y = X.pop("task_type").to_frame()
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, stratify=Y, test_size=0.20, random_state=42
    )
    X_test, X_val, y_test, y_val = train_test_split(
        X_test, y_test, stratify=y_test, test_size=0.50, random_state=42
    )
    pd.concat([X_train, y_train], axis=1).to_csv("train.csv")
    pd.concat([X_test, y_test], axis=1).to_csv("test.csv")
    pd.concat([X_val, y_val], axis=1).to_csv("validation.csv")


read_sinhala_data(
    os.path.join(sinhala_root, "Sinhala_Data.csv"),
    os.path.join(sinhala_root, "Sinhala_Sentences.csv"),
    os.path.join(sinhala_root, "export.csv"),
)


dir_dict = {
    "train": "train.csv",
    "valid": "validation.csv",
    "test": "test.csv",
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
        transcript_df = pd.read_csv(os.path.join(sinhala_root, "data", dir_dict[x]))
        # lines = sorted(transcript_df.values, key=lambda s: s[0])
        for row in transcript_df.values:
            words = row[4].replace(" ", "_") + " " + " ".join([ch for ch in row[3]])
            path_arr = row[1].split("/")
            utt_id = path_arr[-2] + "_" + path_arr[-1]
            text_f.write(utt_id + " " + words + "\n")
            wav_scp_f.write(utt_id + " " + hyper_root + "/" + row[1] + "\n")
            utt2spk_f.write(utt_id + " " + row[2] + "\n")

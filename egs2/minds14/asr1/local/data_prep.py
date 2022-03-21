# Copyright 2021  Young Min Kim
#           2021  Carnegie Mellon University
# Apache 2.0

import os
import re
import sys

import pandas as pd
from sklearn.model_selection import train_test_split


if len(sys.argv) != 3:
    print("Usage: python data_prep.py [MINDS14_DIR] [LANG]")
    sys.exit(1)


MINDS14_DIR = sys.argv[1]  # raw dataset zip file contents should be here
LANG = sys.argv[2] # language
DATA_DIR = "data"  # processed data should go here


def read_minds14_data(dataset_csv, export_csv):
    dataset_df = pd.read_csv(dataset_csv)
    output_df = []

    for i in range(len(dataset_df)):
        wav_name, intent_name, transcript = (
            dataset_df.iloc[i]["filepath"],
            dataset_df.iloc[i]["intent"],
            dataset_df.iloc[i]["text_asr"],
        )

        # clean transcript
        # export audio of for the crop with wav_path_start_duration
        export_path = os.path.join(MINDS14_DIR, "audio", wav_name)
        # Append to output_df
        output_df.append(
            [
                export_path,
                "unknown",
                transcript,
                intent_name,
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

    pd.concat([X_train, y_train], axis=1).to_csv(os.path.join(DATA_DIR, "train.csv"))
    pd.concat([X_test, y_test], axis=1).to_csv(os.path.join(DATA_DIR, "test.csv"))
    pd.concat([X_val, y_val], axis=1).to_csv(os.path.join(DATA_DIR, "validation.csv"))


read_minds14_data(
    os.path.join(MINDS14_DIR, f"text/{LANG}.csv"),  # ~/text/lang-LOCALE.csv
    os.path.join(MINDS14_DIR, "export.csv")
)


dir_dict = {
    "train": "train.csv",
    "valid": "validation.csv",
    "test": "test.csv",
}


for partition in dir_dict:
    with open(os.path.join(DATA_DIR, partition, "text"), "w", encoding='utf-8') as text_f, \
         open(os.path.join(DATA_DIR, partition, "wav.scp"), "w", encoding='utf-8') as wav_scp_f, \
         open(os.path.join(DATA_DIR, partition, "transcript"), "w", encoding='utf-8') as transcript_f, \
         open(os.path.join(DATA_DIR, partition, "utt2spk"), "w", encoding='utf-8') as utt2spk_f:

        text_f.truncate()
        wav_scp_f.truncate()
        utt2spk_f.truncate()

        transcript_df = pd.read_csv(os.path.join(DATA_DIR, dir_dict[partition]))
        for row in transcript_df.values:
            words = row[4] + " " + row[3]
            path_arr = row[1].split("/")
            utt_id = path_arr[-2] + "_" + path_arr[-1]
            text_f.write(f"{utt_id} {words}\n")
            wav_scp_f.write(utt_id + " " + row[1] + "\n")
            utt2spk_f.write(utt_id + " " + row[2] + "\n")

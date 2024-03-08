#!/usr/bin/env python3

import json
import os
import re
import sys

import pandas as pd
import scipy
from scipy.io import wavfile

if len(sys.argv) != 2:
    print("Usage: python data_prep.py [root]")
    sys.exit(1)
root = sys.argv[1]


def save_json(fname, dict_name):
    with open(fname, "w") as f:
        f.write(json.dumps(dict_name, indent=4))


dir_dict = {
    "devel": "slue-sqa5_dev.tsv",
    "test": "slue-sqa5_test.tsv",
}

timestamp_dir = {
    "devel": "dev/word2time",
    "test": "test/word2time",
}

for x in dir_dict:
    transcript_df = pd.read_csv(os.path.join(root, dir_dict[x]), sep="\t")
    utt_dict = {}
    for row in transcript_df.values:
        if "\t" in row[1]:
            row = [row[0]] + row[1].split("\t") + list(row[2:])
        uttid = row[3] + "_" + row[0] + "_" + row[4]
        speaker = row[3]
        if x == "devel":
            wav = 'dev/question/"' + row[0] + '".wav'
            sample_rate, data = wavfile.read(os.path.join(root, wav.replace('"', "")))
            quest_t = len(data) / sample_rate
            wav += ' dev/document/"' + row[4] + '".wav'
        else:
            wav = 'test/question/"' + row[0] + '".wav'
            sample_rate, data = wavfile.read(os.path.join(root, wav.replace('"', "")))
            quest_t = len(data) / sample_rate
            wav += ' test/document/"' + row[4] + '".wav'

        question_transcript = row[2].lower()
        try:
            doc_transcript = row[6].lower()
        except:
            row = list(row[:5]) + list(row[5].split("\t"))
        timestamp_file = open(
            os.path.join(os.path.join(root, timestamp_dir[x]), row[4] + ".txt")
        )
        timestamp_line_arr = [line for line in timestamp_file]
        doc_sentence = ""
        found_ever = 0
        unique_start_dict = {}
        unique_end_dict = {}
        for val in row[9].split("], ["):
            start_time = float(val.split('"')[-1].split(", ")[1].strip())
            if start_time not in unique_start_dict:
                unique_start_dict[start_time] = 1
            else:
                continue
            end_time = float(val.split('"')[-1].split(",")[2].strip().replace("]", ""))
            start_time = start_time + quest_t
            end_time = end_time + quest_t
            ans_word = val.split(",")[0].replace("[", "").replace('"', "").lower()
            if uttid not in utt_dict:
                utt_dict[uttid] = [[ans_word, start_time, end_time]]
            else:
                utt_dict[uttid].append([ans_word, start_time, end_time])
        save_json("data/" + x + "/timestamp", utt_dict)

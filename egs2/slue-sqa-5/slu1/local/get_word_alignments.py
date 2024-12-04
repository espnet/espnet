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
        except AttributeError:
            row = list(row[:5]) + list(row[5].split("\t"))
        timestamp_file = open(
            os.path.join(os.path.join(root, timestamp_dir[x]), row[4] + ".txt")
        )
        timestamp_line_arr = [line for line in timestamp_file]
        utt_dict[uttid] = []
        while len(timestamp_line_arr[0].split("\t")) == 1:
            timestamp_line_arr = timestamp_line_arr[1:]
        a1 = float(timestamp_line_arr[0].split("\t")[2])
        utt_dict[uttid].append(["", 0.0, quest_t + a1])
        continue_start = False
        continue_start_t = 0
        for line_id in range(len(timestamp_line_arr)):
            line = timestamp_line_arr[line_id]
            if len(line.rstrip().split("\t")) == 4:
                word = line.rstrip().split("\t")[1]
                start_t = quest_t + float(line.rstrip().split("\t")[2])
                end_t = quest_t + float(line.rstrip().split("\t")[3])
            else:
                word = '""'
                if continue_start:
                    start_t = continue_start_t
                else:
                    start_t = quest_t + float(
                        timestamp_line_arr[line_id - 1].rstrip().split("\t")[3]
                    )
                if line_id == len(timestamp_line_arr) - 1:
                    continue
                if len(timestamp_line_arr[line_id + 1].rstrip().split("\t")) == 1:
                    continue_start = True
                    continue_start_t = start_t
                    continue
                end_t = quest_t + float(
                    timestamp_line_arr[line_id + 1].rstrip().split("\t")[2]
                )
                continue_start = False
            utt_dict[uttid].append([word, start_t, end_t])
        save_json("data/" + x + "/all_word_alignments.json", utt_dict)

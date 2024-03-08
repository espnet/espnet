#!/usr/bin/env python3

import os
import re
import sys

import pandas as pd

if len(sys.argv) != 2:
    print("Usage: python data_prep.py [root]")
    sys.exit(1)
root = sys.argv[1]

dir_dict = {
    "train": "slue-sqa5_fine-tune.tsv",
    "devel": "slue-sqa5_dev.tsv",
    "test": "slue-sqa5_test.tsv",
}

timestamp_dir = {
    "train": "fine-tune/word2time",
    "devel": "dev/word2time",
    "test": "test/word2time",
}


for x in dir_dict:
    with open(os.path.join("data", x, "text"), "w") as text_f, open(
        os.path.join("data", x, "wav.scp"), "w"
    ) as wav_scp_f, open(os.path.join("data", x, "utt2spk"), "w") as utt2spk_f:
        text_f.truncate()
        wav_scp_f.truncate()
        utt2spk_f.truncate()
        transcript_df = pd.read_csv(os.path.join(root, dir_dict[x]), sep="\t")
        for row in transcript_df.values:
            if "\t" in row[1]:
                row = [row[0]] + row[1].split("\t") + list(row[2:])
            uttid = row[3] + "_" + row[0] + "_" + row[4]
            speaker = row[3]
            if x == "train":
                wav = 'fine-tune/question/"' + row[0] + '".wav'
                wav += ' fine-tune/document/"' + row[4] + '".wav'
            elif x == "devel":
                wav = 'dev/question/"' + row[0] + '".wav'
                wav += ' dev/document/"' + row[4] + '".wav'
            else:
                wav = 'test/question/"' + row[0] + '".wav'
                wav += ' test/document/"' + row[4] + '".wav'

            question_transcript = row[2].lower()
            try:
                doc_transcript = row[6].lower()
            except:
                row = list(row[:5]) + list(row[5].split("\t"))
            if len(row[9].split("], [")) > 1:
                timestamp_file = open(
                    os.path.join(os.path.join(root, timestamp_dir[x]), row[4] + ".txt")
                )
                timestamp_line_arr = [line for line in timestamp_file]
                doc_sentence = ""
                found_ever = 0
                unique_start_dict = {}
                unique_end_dict = {}
                for val in row[9].split("], ["):
                    if (
                        val.split('"')[-1].split(", ")[1].strip()
                        not in unique_start_dict
                    ):
                        unique_start_dict[val.split(", ")[1].strip()] = 1
                    if (
                        val.split('"')[-1].split(",")[2].strip().replace("]", "")
                        not in unique_end_dict
                    ):
                        unique_end_dict[val.split(", ")[2].strip().replace("]", "")] = 1
                for line in timestamp_line_arr:
                    found_start = False
                    found_end = False
                    if len(line.split("\t")) == 1:
                        continue
                    if line.split("\t")[2] in unique_start_dict:
                        found_start = True
                    if line.split("\t")[3].strip() in unique_end_dict:
                        found_end = True
                        found_ever += 1
                    if found_start:
                        doc_sentence += "ANS "
                    doc_sentence += line.split("\t")[1] + " "
                    if found_end:
                        doc_sentence += "ANS "
                assert found_ever == len(unique_start_dict)
            else:
                timestamp_file = open(
                    os.path.join(os.path.join(root, timestamp_dir[x]), row[4] + ".txt")
                )
                timestamp_line_arr = [line for line in timestamp_file]
                ans_word = row[9].split(",")[0].replace("[", "").replace('"', "")
                doc_sentence = ""
                found_ever = 0
                for line in timestamp_line_arr:
                    found_start = False
                    found_end = False
                    if len(line.split("\t")) == 1:
                        continue
                    if (
                        line.split("\t")[2]
                        == row[9].split('"')[-1].split(", ")[1].strip()
                    ):
                        found_start = True
                    if line.split("\t")[3].strip() == row[9].split('"')[-1].split(", ")[
                        2
                    ].strip().replace("]", ""):
                        found_end = True
                        found_ever += 1
                    if found_start:
                        doc_sentence += "ANS "
                    doc_sentence += line.split("\t")[1] + " "
                    if found_end:
                        doc_sentence += "ANS "
                assert found_ever == 1
            doc_transcript = doc_sentence.strip()

            words = question_transcript + " SEP " + doc_transcript
            if "\n" in words:
                import pdb

                pdb.set_trace()
            text_f.write("{} {}\n".format(uttid, words))
            utt2spk_f.write("{} {}\n".format(uttid, speaker))
            wav_scp_f.write(
                f"{uttid} sox {os.path.join(root,wav.split()[0])} {os.path.join(root,wav.split()[1])} -t wav -r 16k - |\n"
            )

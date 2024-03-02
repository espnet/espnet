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
    "train": "slue-ted_train/slue-ted_train.tsv",
    "devel": "slue-ted_dev/slue-ted_dev.tsv",
    "test": "slue-ted_test/slue-ted_test.tsv",
}


for x in dir_dict:
    with open(os.path.join("data", x, "text"), "w") as text_f, open(
        os.path.join("data", x, "wav.scp"), "w"
    ) as wav_scp_f, open(os.path.join("data", x, "utt2spk"), "w") as utt2spk_f, open(
        os.path.join("data", x, "segments"), "w"
    ) as segment_f:
        text_f.truncate()
        wav_scp_f.truncate()
        utt2spk_f.truncate()
        transcript_df = pd.read_csv(os.path.join(root, dir_dict[x]), sep="\t")
        for row in transcript_df.values:
            segmentid = str(row[0])
            speaker = row[2]
            if "-and-" in speaker:
                speaker = "Two_Speaker_" + speaker.strip().replace(" ", "-")
            elif "-+-" in speaker:
                speaker = "Two_Speaker_" + speaker.strip().replace(" ", "-")
            elif "-with-" in speaker:
                speaker = "Two_Speaker_" + speaker.strip().replace(" ", "-")
            elif ",-" in speaker:
                speaker = "Two_Speaker_" + speaker.strip().replace(" ", "-")
            else:
                speaker = "Speaker_" + speaker.strip().replace(" ", "-")
            uttid = speaker + "_" + str(row[0]) + "_000000"
            words = (
                row[4]
                .strip()
                .encode("ascii", "ignore")
                .decode()
                .replace("\n", " ")
                .lower()
                + " [sep] "
                + row[5]
                .strip()
                .encode("ascii", "ignore")
                .decode()
                .replace("\n", " ")
                .lower()
            )
            words = words.replace("\r", " ")
            text_f.write("{} {}\n".format(uttid, words))
            utt2spk_f.write("{} {}\n".format(uttid, speaker))
            if x == "devel":
                path = os.path.join(root, "slue-ted_dev/dev", segmentid + ".flac")
            else:
                path = os.path.join(
                    root, "slue-ted_" + x + "/" + x, segmentid + ".flac"
                )
            wav_scp_f.write(f"{segmentid} {path}\n")
            segment_f.write("{} {} 0 30\n".format(uttid, segmentid))

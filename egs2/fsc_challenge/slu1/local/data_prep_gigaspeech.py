# Copyright 2021  Siddhant Arora
#           2021  Carnegie Mellon University
# Apache 2.0


import os
import re
import string
import sys

import pandas as pd

if len(sys.argv) != 2:
    print("Usage: python data_prep.py [fsc_root]")
    sys.exit(1)
fsc_root = sys.argv[1]

dir_dict = {
    "train": "train_data.csv",
    "valid": "valid_data.csv",
    "utt_test": "utterance_test_data.csv",
    "spk_test": "speaker_test_data.csv",
}


for x in dir_dict:
    # Download gigaspeech ASR model https://zenodo.org/record/4630406 to exp
    # and then run inference on FSC dataset
    asr_transcript_file = open(
        "exp/asr_train_asr_raw_en_bpe5000/inference_asr_model_valid.acc.ave_10best/"
        + x
        + "/score_wer/hyp.trn",
        "r",
    )
    transcript_arr = [line for line in asr_transcript_file]
    transcript_dict = {}
    for line in transcript_arr:
        wav_name = "-".join(line.split("\t")[1].strip()[1:-1].split("-")[1:])
        transcript = " ".join(line.split("\t")[0].split()).lower()
        if transcript == "":
            transcript = "blank"
        print(wav_name)
        print(transcript)
        transcript_dict[wav_name] = transcript
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
        transcript_f.truncate()
        transcript_df = pd.read_csv(
            os.path.join(fsc_root, "data/challenge_splits", dir_dict[x])
        )
        # lines = sorted(transcript_df.values, key=lambda s: s[0])
        for row in transcript_df.values:
            words = (
                row[5].replace(" ", "_")
                + "_"
                + row[6].replace(" ", "_")
                + "_"
                + row[7].replace(" ", "_")
                + " "
                + row[4]
                .encode("ascii", "ignore")
                .decode()
                .lower()
                .translate(str.maketrans("", "", string.punctuation))
            )
            print(words)
            path_arr = row[2].split("/")
            utt_id = path_arr[-2] + "_" + path_arr[-1]
            # print(utt_id + " " + words + "\n")
            text_f.write(utt_id + " " + words + "\n")
            wav_scp_f.write(utt_id + " " + fsc_root + "/" + row[2] + "\n")
            utt2spk_f.write(utt_id + " " + row[3] + "\n")
            transcript_f.write(utt_id + " " + transcript_dict[utt_id] + "\n")

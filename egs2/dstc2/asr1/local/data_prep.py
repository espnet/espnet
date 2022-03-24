#!/usr/bin/env bash

# Copyright 2021  Siddhant Arora
#           2021  Carnegie Mellon University
# Apache 2.0


import os
import re
import sys
import pandas as pd

if len(sys.argv) != 3:
    print("Usage: python data_prep.py [dstc2_root] coverage")
    sys.exit(1)
dstc2_root = sys.argv[1]
coverage = int(sys.argv[2])

dir_dict = {
    "train": "train.csv",
    "valid": "validation.csv",
    "test": "test.csv",
}

#specify the coverage 

def _get_stratified_sampled_data(data,coverage):
    total_sample_count = data.shape[0]

    data["labels_tuple"] = data.dialog_acts.apply(lambda x: tuple(x))   
    unique_data = data.drop_duplicates(subset=['labels_tuple'], keep='first')

    
    unique_sample_count = unique_data.shape[0]

    print("coverage",coverage)
    print("total_sample_count",total_sample_count)
    print("unique_sample_count",unique_sample_count)
    
    rem_sample_count = int(np.round(abs((float(coverage)*total_sample_count) - unique_sample_count)))
    data = data[~data.isin(unique_data)].dropna()

    print("rem_sample_count",rem_sample_count)
    
    rem_data = data.sample(n = rem_sample_count, random_state = 42).reset_index(drop=True)
    
    sampled_data = pd.concat([unique_data, rem_data], ignore_index=True)

    
    print("sampled_data",sampled_data.shape)
    return sampled_data


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
        transcript_df = pd.read_csv(os.path.join(dstc2_root, "data", dir_dict[x]))
        
        if x==train and coverage<1:
            transcript_df = _get_stratified_sampled_data(transcript_df,coverage)
            
        transcript_df  = transcript_df.reset_index()  # make sure indexes pair with number of rows
        
        for index, row in transcript_df.iterrows():
            words = (
                " <sep> ".join(sorted(eval(row["dialog_acts"])))
                + " <utt> "
                + row["transcript"].replace("<unk>", "unk").encode("ascii", "ignore").decode()
            )
            path_arr = row["audio_file_path"].split("/")
            utt_id = path_arr["audio_file_path"] + "_" + path_arr[-1]
            speaker_id = path_arr[-2]
            text_f.write(utt_id + " " + words + "\n")
            wav_scp_f.write(utt_id + " " + dstc2_root + "/" + row["audio_file_path"] + "\n")
            utt2spk_f.write(utt_id + " " + speaker_id + "\n")

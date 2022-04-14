"""
Parse the data
"""
import sys
import os
import pandas as pd
import numpy as np
import json

train_size = 0.6
dev_size = 0.2
random_state = 11737

data_dir = sys.argv[1]  # downloads/TAT-Vol2
output_text = sys.argv[2]
# han-tw, tailo, tailo-num, poj
# 漢羅台文 (Taiwanese Han characters) 台羅 (tailo) 台羅數字調 (tailo, number tones) 白話字 (POJ)
spk_id = "spk001"

text_column_name = {
    "han-tw": "漢羅台文",
    "tailo": "台羅",
    "tailo-num": "台羅數字調",
    "poj": "白話字"
}[output_text]

data_dicts = []
for speaker in os.listdir(os.path.join(data_dir, "json")):
    # ignore .DS_Store
    if os.path.isdir(os.path.join(data_dir, "json", speaker)):
        for utterance_f in os.listdir(os.path.join(data_dir, "json", speaker)):
            with open(os.path.join(data_dir, "json", speaker, utterance_f)) as f:
                info = json.load(f)
                utt_id = info["提示卡編號"] + '-' + info['句編號']
                data_dicts.append({
                    "utterance": utt_id, # ex: 0035-5.23
                    "speaker": info["發音人"],
                    "path": os.path.join(data_dir, "condenser", speaker, utt_id + '-03.wav'),
                    "transcription": info[text_column_name]
                })
train_df = pd.DataFrame(data_dicts)

# goal: CSV with utterance, speaker, filepath, transcription

def create_files(df, directory):
    text_lines, scp_lines, utt2spk_lines = [], [], []

    for idx, row in df.iterrows():
        wav_file_path = row["path"]
        # add speaker-ids prefixes of utt-ids
        spk_id = row["speaker"]
        utt_id = spk_id + "_" + row["utterance"]
        transcription = row["transcription"]

        # utterance id and transcription
        text_lines.append(f"{utt_id} {transcription}\n")
        # utterance id and wave file path 
        scp_lines.append(f"{utt_id} {wav_file_path}\n")
        # utterance id and speaker id
        utt2spk_lines.append(f"{utt_id} {spk_id}\n")

    # sort
    text_lines.sort()
    scp_lines.sort()
    utt2spk_lines.sort()

    # write to file
    with open(f"{directory}/text", "w+") as text_file:
        text_file.writelines(text_lines)

    with open(f"{directory}/wav.scp", "w+") as scp_file:
        scp_file.writelines(scp_lines)

    with open(f"{directory}/utt2spk", "w+") as utt2spk_file:
        utt2spk_file.writelines(utt2spk_lines)


print("Creating files for train...", end="")
create_files(train_df, "data/train")
print("Done.")

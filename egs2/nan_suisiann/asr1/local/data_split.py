"""
Split data to train, dev, test
"""
import sys
import os
import pandas as pd
import numpy as np

train_size = 0.6
dev_size = 0.2
random_state = 11737

data_dir = sys.argv[1]  # downloads/0.2.1
output_text = sys.argv[2] # tailo or cmn
spk_id = "spk001"

if output_text == "tailo":
    text_column_name = "羅馬字"
elif output_text == "cmn":
    text_column_name = "漢字"
else:
    raise NotImplementedError

df = pd.read_csv(os.path.join(data_dir, "SuiSiann.csv"))


train_df, dev_df, test_df = \
              np.split(df.sample(frac=1, random_state=random_state), 
                       [int(train_size*len(df)), 
                       int((train_size+dev_size)*len(df))])


print(
    f"# train: {len(train_df)}, # dev:{len(dev_df)}, # test:{len(test_df)}"
)

def create_files(df, directory):
    text_lines, scp_lines, utt2spk_lines = [], [], []

    for idx, row in df.iterrows():
        wav_file_path = os.path.join(data_dir, row["音檔"])
        utt_id = row["音檔"].split("/")[-1].split(".")[0]
        # add speaker-ids prefixes of utt-ids
        utt_id = spk_id+utt_id
        transcription = row[text_column_name]

        text_lines.append(f"{utt_id} {transcription}\n")

        scp_lines.append(f"{utt_id} {wav_file_path}\n")

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

print("Creating files for dev...", end="")
create_files(dev_df, "data/dev")
print("Done.")

print("Creating files for test...", end="")
create_files(test_df, "data/test")
print("Done.")

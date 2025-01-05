import os
import random
import sys
from pathlib import Path

import pandas as pd

DATA_READ_ROOT = sys.argv[1]
DATA_WRITE_ROOT = sys.argv[2]

random.seed(42)
df = pd.read_csv(os.path.join(DATA_READ_ROOT, "train.csv"))
all_recordist = df.recordist.unique()
random.shuffle(all_recordist)
train_recordist = set(all_recordist[: int(0.6 * len(all_recordist))])
valid_recordist = set(
    all_recordist[int(0.6 * len(all_recordist)) : int(0.8 * len(all_recordist))]
)
test_recordist = set(all_recordist[int(0.8 * len(all_recordist)) :])


def convert(row):
    if row["recordist"] in train_recordist:
        split = "train"
    elif row["recordist"] in valid_recordist:
        split = "valid"
    else:
        split = "test"
    filepath = os.path.join(
        DATA_READ_ROOT, "train_audio", row["ebird_code"], row["filename"]
    )

    new_row = pd.Series(
        {
            "path": filepath,
            "label": str(row["ebird_code"]).replace(" ", "_"),
            "split": split,
        }
    )
    return new_row


df = df.apply(convert, axis=1)

split2df = {
    "cbi.dev": df[df.split == "valid"],
    "cbi.train": df[df.split == "train"],
    "cbi.test": df[df.split == "test"],
}
for split, df in split2df.items():
    text_path = os.path.join(DATA_WRITE_ROOT, split, "text")
    wav_path = os.path.join(DATA_WRITE_ROOT, split, "wav.scp")
    utt2spk_path = os.path.join(DATA_WRITE_ROOT, split, "utt2spk")
    os.makedirs(os.path.dirname(text_path), exist_ok=True)
    with open(text_path, "w") as text_f, open(wav_path, "w") as wav_f, open(
        utt2spk_path, "w"
    ) as utt2spk_f:
        for index, row in df.iterrows():
            uttid = f"{split}-{index}"
            print(f"{uttid} {row['path']}", file=wav_f)
            print(f"{uttid} {row['label']}", file=text_f)
            print(f"{uttid} dummy", file=utt2spk_f)

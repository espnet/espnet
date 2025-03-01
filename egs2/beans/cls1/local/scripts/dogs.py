import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_READ_ROOT = sys.argv[1]
DATA_WRITE_ROOT = sys.argv[2]

df = pd.read_csv(os.path.join(DATA_READ_ROOT, "annotations.csv"))


def convert(row):
    filepath = os.path.join(DATA_READ_ROOT, "audio", f"{row['filename']}")
    new_row = pd.Series({"path": filepath, "label": str(row["name"]).replace(" ", "_")})
    return new_row


df = df.apply(convert, axis=1)

df_train, df_valid_test = train_test_split(
    df, test_size=0.4, random_state=42, shuffle=True, stratify=df["label"]
)
df_valid, df_test = train_test_split(
    df_valid_test,
    test_size=0.5,
    random_state=42,
    shuffle=True,
    stratify=df_valid_test["label"],
)
df_train_low, _ = train_test_split(
    df_train, test_size=0.8, random_state=42, shuffle=True, stratify=df_train["label"]
)

df_train = df_train.sort_index()
df_train_low = df_train_low.sort_index()
df_valid = df_valid.sort_index()
df_test = df_test.sort_index()

split2df = {
    "dogs.dev": df_valid,
    "dogs.train": df_train,
    "dogs.train-low": df_train_low,
    "dogs.test": df_test,
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

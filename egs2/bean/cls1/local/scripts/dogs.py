import sys
import pandas as pd
from plumbum import local, FG
from pathlib import Path
from sklearn.model_selection import train_test_split

local["mkdir"]["-p", "data/dogs/wav"]()
(
    local["wget"][
        "-O",
        "data/dogs/dog_barks.zip",
        "https://storage.googleapis.com/ml-bioacoustics-datasets/dog_barks.zip",
    ]
    & FG
)
local["unzip"]["data/dogs/dog_barks.zip", "-d", "data/dogs/"] & FG

df = pd.read_csv("data/dogs/annotations.csv")


def convert(row):
    src_path = Path("data/dogs/audio") / row["filename"]
    tgt_path = Path("data/dogs/wav") / (Path(row["filename"]).stem + ".wav")

    print(f"Converting {src_path} ...", file=sys.stderr)

    local["sox"][src_path, "-r 44100", "-R", tgt_path]()

    new_row = pd.Series({"path": tgt_path, "label": row["name"]})

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

df_train.to_csv("data/dogs/annotations.train.csv")
df_train_low.to_csv("data/dogs/annotations.train-low.csv")
df_valid.to_csv("data/dogs/annotations.valid.csv")
df_test.to_csv("data/dogs/annotations.test.csv")

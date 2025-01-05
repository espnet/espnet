import numpy as np
import pandas as pd
from plumbum import FG, local
from sklearn.model_selection import train_test_split

(
    local["git"][
        "clone", "https://github.com/HumBug-Mosquito/HumBugDB.git", "data/HumBugDB"
    ]
    & FG
)

for i in [1, 2, 3, 4]:
    (
        local["wget"][
            "-O",
            f"data/HumBugDB/humbugdb_neurips_2021_{i}.zip",
            f"https://zenodo.org/record/4904800/files/humbugdb_neurips_2021_{i}.zip?download=1",
        ]
        & FG
    )
    (
        local["unzip"][
            f"data/HumBugDB/humbugdb_neurips_2021_{i}.zip",
            "-d",
            "data/HumBugDB/data/audio/",
        ]
        & FG
    )


df = pd.read_csv("data/HumBugDB/data/metadata/neurips_2021_zenodo_0_0_1.csv")
df.species = df.species.fillna("non-mosquito")

# collapse infrequent labels to "others"
value_counts = df.species.value_counts()
to_remove = value_counts[value_counts <= 100].index
df.species.replace(to_remove, "others", inplace=True)


def convert(row):
    new_row = pd.Series(
        {"path": f"data/HumBugDB/data/audio/{row['id']}.wav", "label": row["species"]}
    )

    return new_row


df = df.apply(convert, axis=1)

# split to train:valid:test = 6:2:2
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

df_train.to_csv("data/HumBugDB/data/metadata/train.csv")
df_train_low.to_csv("data/HumBugDB/data/metadata/train-low.csv")
df_valid.to_csv("data/HumBugDB/data/metadata/valid.csv")
df_test.to_csv("data/HumBugDB/data/metadata/test.csv")

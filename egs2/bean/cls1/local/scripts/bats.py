import pandas as pd
from plumbum import FG, local
from sklearn.model_selection import train_test_split

local["mkdir"]["-p", "data/egyptian_fruit_bats"]()
(
    local["wget"][
        "https://storage.googleapis.com/ml-bioacoustics-datasets/egyptian_fruit_bats.zip",
        "-P",
        "data/egyptian_fruit_bats",
    ]
    & FG
)
(
    local["unzip"][
        "data/egyptian_fruit_bats/egyptian_fruit_bats.zip",
        "-d",
        "data/egyptian_fruit_bats/",
    ]
    & FG
)

df = pd.read_csv("data/egyptian_fruit_bats/annotations.csv")
df = df.apply(
    lambda r: pd.Series(
        {
            "path": f"data/egyptian_fruit_bats/audio/{r['File Name']}",
            "label": r["Emitter"],
        }
    ),
    axis=1,
)

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

df_train.to_csv("data/egyptian_fruit_bats/annotations.train.csv")
df_train_low.to_csv("data/egyptian_fruit_bats/annotations.train-low.csv")
df_valid.to_csv("data/egyptian_fruit_bats/annotations.valid.csv")
df_test.to_csv("data/egyptian_fruit_bats/annotations.test.csv")

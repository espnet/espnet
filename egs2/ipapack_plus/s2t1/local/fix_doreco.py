from pathlib import Path

import pandas as pd
from data_prep import df_to_kaldi


def get_correct_path(old_path):
    lang = old_path.split("/wds/")[1].split("/")[0]
    return f"downloads/doreco/{lang}{old_path}"


if __name__ == "__main__":
    df = pd.read_csv("downloads/transcript_normalized.csv")
    df = df[df["dataset"] == "doreco"]
    # turn previous hardcoded path to a new cleaner path
    df["path"] = df.apply(lambda row: get_correct_path(row["path"]), axis=1)
    df.to_csv("downloads/transcript_doreco.csv", index=False)
    df_to_kaldi(df, Path("downloads"), Path("data"))

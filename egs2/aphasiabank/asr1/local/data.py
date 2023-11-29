import pandas as pd

# Load on import
df = pd.read_csv("data/split.csv")

lang2lang_id = {
    "English": "EN",
    "French": "FR",
}

train_spks = df[df["split"] == "train"]["spk"].tolist()
test_spks = df[df["split"] == "test"]["spk"].tolist()
val_spks = df[df["split"] == "val"]["spk"].tolist()

# Load mappings
spk2aphasia_label = {}
spk2lang_id = {}
spk2severity = {}
for i, row in df.iterrows():
    spk = row["spk"]
    tag = row["aph_tag"]
    lang_id = row["lang"]
    severity = row["severity"]

    assert tag is not None and lang_id is not None

    spk2aphasia_label[spk] = tag
    spk2lang_id[spk] = lang_id

    if severity is not None:
        spk2severity[spk] = severity

# Create inverse mappings
aph2spks = {}
for spk, aph in spk2aphasia_label.items():
    aph2spks.setdefault(aph, []).append(spk)

severity2spks = {}
for spk, severity in spk2severity.items():
    severity2spks.setdefault(severity, []).append(spk)

lang_id2spks = {}
for spk, lid in spk2lang_id.items():
    lang_id2spks.setdefault(lid, []).append(spk)


def get_utt(spk: str, start_time: int, end_time: int) -> str:
    return f"{spk}-{start_time}_{end_time}"


def utt2time(utt: str) -> (int, int):
    _, timestamps = utt.split("-")
    start, end = timestamps.split("_")
    return int(start), int(end)


def utt2spk(utt: str) -> str:
    return utt.split("-")[-2]

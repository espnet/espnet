#!/usr/bin/env python3

import os
import traceback

import soundfile as sf

try:
    import datasets
except Exception:
    traceback.print_exc()
    print("Error importing datasets library")
    print("datasets can be installed via espnet/tools/installers/install_datasets")
    exit()


LID_MAP = {
    "org_jpn": "jpn",
    "lga": "lug",
    "ory": "ori",
    "azj": "aze",
    "arb": "ara",
    "tgl": "fil",
    "swh": "swa",
}
PASS_LANGS = {"nno", "nob", "nor"}
MS_SPEECH_LIDS = {"tam", "tel", "guj"}


def normalize_lid(split, uttid, lid):
    lid = lid.strip()

    if split == "dev_dialect" and uttid.startswith("ms_speech_"):
        parts = uttid.split("_")
        if len(parts) >= 3 and parts[2] in MS_SPEECH_LIDS:
            return parts[2]

    lid = LID_MAP.get(lid, lid)

    if split in {"train", "dev"} and lid in PASS_LANGS:
        return None

    return lid


def replace_lid(text, lid):
    text = text.strip()
    if text.startswith("[") and "]" in text:
        text = text.split("]", 1)[1].strip()
    return f"[{lid}] {text}"


def save_audio_to_disk(sample):
    sf.write(f"data/raw_audio/{sample['id']}.wav", sample["audio"]["array"], 16000)
    return sample


ds = datasets.load_dataset(
    "espnet/ml_superb_hf",
    cache_dir=os.environ.get("MLSUPERB2_HF_CACHE", "."),
)

train_text_out = open("data/train/text", "w")
train_wav_out = open("data/train/wav.scp", "w")
train_utt_out = open("data/train/utt2spk", "w")

dev_text_out = open("data/dev/text", "w")
dev_wav_out = open("data/dev/wav.scp", "w")
dev_utt_out = open("data/dev/utt2spk", "w")

dialect_text_out = open("data/dev_dialect/text", "w")
dialect_wav_out = open("data/dev_dialect/wav.scp", "w")
dialect_utt_out = open("data/dev_dialect/utt2spk", "w")

nlsyms_out = open("data/local/nlsyms.txt", "w")

train_lids = set()

for split, text_out, wav_out, utt_out in [
    ("train", train_text_out, train_wav_out, train_utt_out),
    ("dev", dev_text_out, dev_wav_out, dev_utt_out),
    ("dev_dialect", dialect_text_out, dialect_wav_out, dialect_utt_out),
]:
    texts = ds[split]["text"]
    ids = ds[split]["id"]
    lids = ds[split]["language"]

    for idx, text, raw_lid in zip(ids, texts, lids):
        lid = normalize_lid(split, idx, raw_lid)
        if lid is None:
            continue
        if split == "train":
            train_lids.add(lid)

        text_out.write(f"{idx} {replace_lid(text, lid)}\n")
        utt_out.write(f"{idx} {idx}\n")
        wav_out.write(f"{idx} data/raw_audio/{idx}.wav\n")

for lid in sorted(train_lids):
    nlsyms_out.write(f"[{lid}]\n")

ds["train"].map(save_audio_to_disk)
ds["dev"].map(save_audio_to_disk)
ds["dev_dialect"].map(save_audio_to_disk)

train_text_out.close()
train_wav_out.close()
train_utt_out.close()
dev_text_out.close()
dev_wav_out.close()
dev_utt_out.close()
dialect_text_out.close()
dialect_wav_out.close()
dialect_utt_out.close()
nlsyms_out.close()

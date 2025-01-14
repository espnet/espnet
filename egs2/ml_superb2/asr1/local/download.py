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

ds = datasets.load_dataset("espnet/ml_superb_hf", cache_dir=".")

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


def save_audio_to_disk(sample):
    sf.write(f"data/raw_audio/{sample['id']}.wav", sample["audio"]["array"], 16000)
    return sample


texts = ds["train"]["text"]
ids = ds["train"]["id"]

for idx, text in zip(ids, texts):
    train_text_out.write(f"{idx} {text.strip().replace('[org_jpn]', '[jpn]')}\n")
    train_utt_out.write(f"{idx} {idx}\n")
    train_wav_out.write(f"{idx} data/raw_audio/{idx}.wav\n")

texts = ds["dev"]["text"]
ids = ds["dev"]["id"]

for idx, text in zip(ids, texts):
    dev_text_out.write(f"{idx} {text.strip().replace('[org_jpn]', '[jpn]')}\n")
    dev_utt_out.write(f"{idx} {idx}\n")
    dev_wav_out.write(f"{idx} data/raw_audio/{idx}.wav\n")

texts = ds["dev_dialect"]["text"]
ids = ds["dev_dialect"]["id"]

for idx, text in zip(ids, texts):
    # replace with dialect lid for ms_speech
    if "ms_speech" in idx:
        lid = idx.split("_")[2]
        text = text.replace("[hin]", f"[{lid}]")
    dialect_text_out.write(f"{idx} {text.strip().replace('[org_jpn]', '[jpn]')}\n")
    dialect_utt_out.write(f"{idx} {idx}\n")
    dialect_wav_out.write(f"{idx} data/raw_audio/{idx}.wav\n")

lids = ds["train"]["language"]
lids = list(set(lids))

for lid in lids:
    if lid != "org_jpn":
        nlsyms_out.write(f"[{lid.strip()}]\n")

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

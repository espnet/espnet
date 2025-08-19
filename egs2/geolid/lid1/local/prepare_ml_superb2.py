import os
import traceback
import sys
import soundfile as sf
from tqdm import tqdm

try:
    import datasets
except Exception:
    traceback.print_exc()
    print("Error importing datasets library")
    print("datasets can be installed via espnet/tools/installers/install_datasets")
    exit()

ds = datasets.load_dataset("espnet/ml_superb_hf")

def save_audio_to_disk(sample, set_name, lang):
    os.makedirs(f"downloads/ml_superb2/{set_name}/{lang}", exist_ok=True)
    save_path = f"downloads/ml_superb2/{set_name}/{lang}/{sample['id']}.wav"
    if not os.path.exists(save_path):
        sf.write(save_path, sample["audio"]["array"], 16000)

set_names = ["train", "dev", "dev_dialect"]
pass_langs = ["nno", "nob", "nor"]
# The ML-SUPERB2 official using the expected language code 
# rather than the code directly in huggingface
lang_to_expect_langs = {
    "lga": "lug",
    "org_jpn": "jpn",
    "ory": "ori",
    "azj": "aze",
    "arb": "ara",
}

for set_name in set_names:
    subset = ds[set_name]

    os.makedirs(f"data/{set_name}_ml_superb2_lang", exist_ok=True)

    wav_scp = []
    utt2spk = []

    for sample in tqdm(subset, total=len(subset)):
        if sample["language"] in pass_langs:
            continue
        idx = sample["id"]
        lang = sample["language"]
        lang = lang.strip()
        if lang in lang_to_expect_langs:
            lang = lang_to_expect_langs[lang]
        uttid = f"{lang}_{idx}"
        save_audio_to_disk(sample, set_name, lang)
        wav_scp.append(f"{uttid} downloads/ml_superb2/{set_name}/{lang}/{idx}.wav\n")
        utt2spk.append(f"{uttid} {lang}\n")

    with open(f"data/{set_name}_ml_superb2_lang/wav.scp", "w") as wav_scp_fp:
        wav_scp_fp.writelines(sorted(wav_scp))
    with open(f"data/{set_name}_ml_superb2_lang/utt2spk", "w") as utt2spk_fp:
        utt2spk_fp.writelines(sorted(utt2spk))

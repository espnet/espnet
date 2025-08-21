import os
import shutil
from typing import List

import pandas as pd

# cs de en es et fi fr hr hu it lt nl pl ro sk sl
lang_to_iso3 = {
    "cs": "ces",
    "de": "deu",
    "en": "eng",
    "es": "spa",
    "et": "est",
    "fi": "fin",
    "fr": "fra",
    "hr": "hrv",
    "hu": "hun",
    "it": "ita",
    "lt": "lit",
    "nl": "nld",
    "pl": "pol",
    "ro": "ron",
    "sk": "slk",
    "sl": "slv",
}


def build_audio_path(utterance_ids: List[str], base_dir: str):
    wav_scp = []
    for utterance_id in utterance_ids:
        year = utterance_id.split("_")[1][:4]
        path = os.path.join(base_dir, year, f"{utterance_id.split('_', 1)[1]}.ogg")
        wav_scp.append(f"{utterance_id} {path}\n")
    return wav_scp


def build_utt2lang(utterance_ids: List[str], lang_folder_name: str):
    utt2lang = []
    for utterance_id in utterance_ids:
        utt2lang.append(f"{utterance_id} {lang_to_iso3[lang_folder_name]}\n")
    return utt2lang


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="downloads/voxpopuli")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_path = args.dataset_path

    target_folder = "data"

    original_folder = f"{dataset_path}/_transcribed_data"

    dataset_name = "voxpopuli"

    # dev is used for both development and test set, training is used for train set
    for split1, split2 in [
        ("train", f"train_{dataset_name}_lang"),
        ("dev", f"dev_{dataset_name}_lang"),
        ("dev", f"test_{dataset_name}_lang"),
    ]:
        wav_scp = []
        utt2lang = []
        for lang_folder_name in lang_to_iso3.keys():
            lang_folder_path = os.path.join(original_folder, lang_folder_name)
            df = pd.read_csv(
                os.path.join(lang_folder_path, f"asr_{split1}.tsv"), sep="\t"
            )
            df = df.iloc[::-1]
            utterance_ids = df["id"].tolist()
            utterance_ids = [
                f"{lang_to_iso3[lang_folder_name]}_{utterance_id}"
                for utterance_id in utterance_ids
            ]
            wav_scp.extend(build_audio_path(utterance_ids, lang_folder_path))
            utt2lang.extend(build_utt2lang(utterance_ids, lang_folder_name))

        # make directory
        os.makedirs(os.path.join(target_folder, split2), exist_ok=True)

        with open(os.path.join(target_folder, split2, "wav.scp"), "w") as f:
            f.writelines(sorted(wav_scp))
        with open(os.path.join(target_folder, split2, "utt2lang"), "w") as f:
            f.writelines(sorted(utt2lang))


if __name__ == "__main__":
    main()

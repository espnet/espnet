import argparse
import csv
import json
import os
import shutil
from collections import defaultdict

import langcodes
import matplotlib.pyplot as plt
import pandas as pd
import regex as re
from langcodes import tag_is_valid
from tqdm import tqdm

"""
Converts IPAPack++ into OWSM's expected format
"""

ASR = "<asr>"
PR = "<pr>"
G2P = "<g2p>"
P2G = "<p2g>"
TEXT_NA = "<na>"
NO_TIME = "<notimestamps>"
SAMPLE_RATE = 16000
LANG = "<LANG>"  # Should be mapping from utt_id to language code
UNK_LANG = "<unk>"
remove_space_lang = ["<cmn>", "<yue>", "<jpn>", "<tha>", "<lao>"]


def get_lang(lang_name):
    try:
        if tag_is_valid(lang_name):
            langcode = langcodes.get(lang_name).to_alpha3()
        else:
            langcode = langcodes.find(lang_name).to_alpha3()
        if langcode == "zho":
            return "<cmn>"
        else:
            return f"<{langcode}>"
    except LookupError:
        print(f"Unknown language: {lang_name}")
        return "<unk>"


def main(root_dir, output_dir):
    # source directories
    ROOT_DATA_DIR = os.path.join(root_dir, "data")
    ROOT_DF_DIR = os.path.join(root_dir, "downloads")
    # target directory
    os.makedirs(output_dir, exist_ok=True)
    # setup
    tasks = ["asr", "g2p", "p2g", "pr"]
    splits_to_process = ["train", "dev"]

    # ========== Generate multitask text files from csv ==========
    print("Generating multitask text files...")
    # Open text files
    texts = {}
    for split in splits_to_process:
        texts[split] = {}
        for filename in ["text", "text.ctc", "text.prev"]:
            texts[split][filename] = open(
                os.path.join(output_dir, split, filename), "w"
            )

    # Write to text files
    ntt = "<notimestamps>"
    with open(os.path.join(ROOT_DF_DIR, "transcript_normalized.csv"), "r") as f:
        reader = csv.DictReader(f)
        next(reader)  # skip header
        for row in reader:
            split = row["split"]
            if split not in splits_to_process:
                continue
            utt_id = row["utt_id"]
            lang = get_lang(row["lang"])
            text = row["text"]
            path = row["path"]
            ipa_panphon = row["ipa_panphon"]
            ipa_panphon_nosup = row["ipa_panphon_nosup"]

            # phone tokens are surrounded by slashes (/p/)
            ipa_panphon = "/" + "//".join(ipa_panphon.split()) + "/"
            ipa_panphon_nosup = "/" + "//".join(ipa_panphon_nosup.split()) + "/"
            if lang in remove_space_lang:
                text = text.replace(" ", "")

            for task in tasks:
                # text: pr, g2p -> pr; asr, p2g -> orthography
                content = ipa_panphon if task in ["pr", "g2p"] else text
                texts[split]["text"].write(
                    f"{utt_id}_{task} {lang}<{task}>{ntt} {content}\n"
                )
                # text.ctc: all tasks -> reduced pr
                texts[split]["text.ctc"].write(f"{utt_id}_{task} {ipa_panphon_nosup}\n")
                # text.prev: pr, asr -> <na>; g2p -> orthography; p2g -> pr
                if task in ["pr", "asr"]:
                    prev_content = "<na>"
                elif task == "g2p":
                    prev_content = text
                else:
                    prev_content = ipa_panphon
                texts[split]["text.prev"].write(f"{utt_id}_{task} {prev_content}\n")

    for split in texts:
        for filename in texts[split]:
            texts[split][filename].close()

    # ========== Get wav.scp and additional files  ==========
    print("Generating wav.scp and additional files...")
    for split in splits_to_process:
        # wav.scp, utt2num_samples: add task name to get unique IDs
        for filename in ["wav.scp", "utt2num_samples"]:
            oldfile = os.path.join(ROOT_DATA_DIR, split, filename)
            newfile = os.path.join(output_dir, split, filename)
            for task in tasks:
                # take the first column which is the uttID
                os.system(
                    (
                        f'awk \'{{ $1 = $1 "_{task}"; print }}\' OFS=" " '
                        f"{oldfile} >> {newfile}"
                    )
                )
        # spk2utt, utt2spk: write first column of wav.scp twice
        for filename in ["spk2utt", "utt2spk"]:
            infile = os.path.join(output_dir, split, "wav.scp")
            outfile = os.path.join(output_dir, split, filename)
            os.system(f"awk '{{print $1, $1}}' {infile} > {outfile}")
        # feats_type, audio_format: copy as is
        for filename in ["feats_type", "audio_format"]:
            oldfile = os.path.join(ROOT_DATA_DIR, split, filename)
            newfile = os.path.join(output_dir, split, filename)
            shutil.copy2(oldfile, newfile)

    # utils/fix_data_dir.sh will help with sorting and filtering text entries
    print("Finish generating train & dev set!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process IPAPack++")
    parser.add_argument("--root_dir", type=str, default=".", help="Root directory")
    parser.add_argument(
        "--output_dir", type=str, default="dump/raw", help="Output directory"
    )
    args = parser.parse_args()

    main(**vars(args))

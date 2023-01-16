#!/usr/bin/env python3

# Copyright 2021  Sujay Suresh Kumar
#           2021  Carnegie Mellon University
#           2022  University of Stuttgart (Pavel Denisov)
# Apache 2.0

import json
import os
import string as string_lib
import sys
from pathlib import Path

if len(sys.argv) != 2:
    print("Usage: python data_prep.py [catslu_root]")
    sys.exit(1)

catslu_root = sys.argv[1]

BLACKLIST_IDS = ["map-df61ee397d015314dfde80255365428b_4b3d3b2f332793052a000014-1"]


catslu_root_path = Path(catslu_root)

catslu_traindev = Path(os.path.join(catslu_root_path, "catslu_traindev", "data"))
catslu_traindev_domain_dirs = [f for f in catslu_traindev.iterdir() if f.is_dir()]

catslu_test = Path(os.path.join(catslu_root_path, "catslu_test", "data"))
catslu_test_domain_dirs = [f for f in catslu_test.iterdir() if f.is_dir()]


def _process_data(data):
    global BLACKLIST_IDS
    wavs = []
    texts = []
    utt2spk = []
    for dialogue in data:
        dialogue_id = dialogue["dlg_id"]
        for utterance in dialogue["utterances"]:
            wav_path = os.path.join(
                domain_dir, "audios/{}.wav".format(utterance["wav_id"])
            )

            utt_id = "{}-{}-{}".format(
                domain_dir.parts[-1], utterance["wav_id"], utterance["utt_id"]
            )

            manual_transcript = utterance["manual_transcript"]
            manual_transcript = manual_transcript.replace("(unknown)", "")
            manual_transcript = manual_transcript.replace("(side)", "")
            manual_transcript = manual_transcript.replace("(dialect)", "")
            manual_transcript = manual_transcript.replace("(robot)", "")
            manual_transcript = manual_transcript.replace("(noise)", "")

            if utt_id in BLACKLIST_IDS or manual_transcript == "":
                continue

            transcript = []

            for semantic in utterance["semantic"]:
                if len(semantic) == 2:
                    transcript.append(f"{semantic[0]} FILL {semantic[1]}")
                elif len(semantic) == 3:
                    transcript.append(f"{semantic[0]}_{semantic[1]} FILL {semantic[2]}")

            transcript.append(manual_transcript)

            if len(transcript) > 0:
                texts.append(f"{utt_id} " + " SEP ".join(transcript) + "\n")
            else:
                texts.append(f"{utt_id} SEP\n")

            wavs.append(f"{utt_id} {wav_path}\n")
            utt2spk.append(f"{utt_id} {domain_dir.name}-{dialogue_id}\n")

    return wavs, texts, utt2spk


data_json = {"train": "train.json", "devel": "development.json"}

for subset in ["train", "devel"]:
    odir = f"data/{subset}"
    os.makedirs(odir, exist_ok=True)

    with open(os.path.join(odir, "wav.scp"), "w") as wav_f, open(
        os.path.join(odir, "utt2spk"), "w"
    ) as utt2spk_f, open(os.path.join(odir, "text"), "w", encoding="utf-8") as text_f:
        for domain_dir in catslu_traindev_domain_dirs:
            with open(os.path.join(domain_dir, data_json[subset])) as fp:
                data = json.load(fp)

            wav, text, utt2spk = _process_data(data)

            wav_f.writelines(wav)
            text_f.writelines(text)
            utt2spk_f.writelines(utt2spk)

for domain_dir in catslu_test_domain_dirs:
    odir = f"data/test_{domain_dir.name}"
    os.makedirs(odir, exist_ok=True)

    with open(os.path.join(odir, "wav.scp"), "w") as wav_f, open(
        os.path.join(odir, "utt2spk"), "w"
    ) as utt2spk_f, open(
        os.path.join(odir, "text"), "w", encoding="utf-8"
    ) as text_f, open(
        os.path.join(domain_dir, "test.json")
    ) as fp:
        data = json.load(fp)

        wav, text, utt2spk = _process_data(data)

        wav_f.writelines(wav)
        text_f.writelines(text)
        utt2spk_f.writelines(utt2spk)

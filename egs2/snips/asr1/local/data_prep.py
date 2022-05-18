#!/usr/bin/env bash

# Copyright 2021  Yuekai Zhang

import argparse
import json

parser = argparse.ArgumentParser(description="Process snips dataset.")
parser.add_argument("--wav_path", type=str, help="file path for audios")
parser.add_argument(
    "--text_f", type=str, default="data/text.trans", help="file path for text"
)
parser.add_argument(
    "--semantics", type=str, default="data/semantics", help="file path for semantics"
)
parser.add_argument(
    "--utt2spk_f", type=str, default="data/utt2spk", help="file path for utt2spk"
)
parser.add_argument(
    "--wavscp_f", type=str, default="data/wav.scp", help="file path for wav.scp"
)
parser.add_argument(
    "--non_linguistic_symbols",
    type=str,
    default="data/non_linguistic_symbols.txt",
    help="non_linguistic_symbols",
)
args = parser.parse_args()

meta = args.wav_path + "/speech_corpus/metadata.json"
dataset = args.wav_path + "/dataset.json"

with open(meta, "r") as meta_f, open(args.text_f, "w") as text_f, open(
    dataset, "r"
) as dataset, open(args.non_linguistic_symbols, "w") as non_linguistic_symbols, open(
    args.wavscp_f, "w"
) as wavscp_f, open(
    args.utt2spk_f, "w"
) as utt2spk_f, open(
    args.semantics, "w"
) as semantics:
    meta_info, dataset = json.load(meta_f), json.load(dataset)
    intents = dataset["intents"]
    for intent, utts in intents.items():
        non_linguistic_symbols.write(f"{intent.upper()}\n")
        for utt in utts["utterances"]:
            utt_text = ""
            utt_semantic = [intent.upper()]
            for partial_text in utt["data"]:
                utt_text += partial_text["text"]
                if "entity" in partial_text:
                    assert "slot_name" in partial_text
                    entity = partial_text["entity"].upper()
                    utt_semantic.append(entity)
                    non_linguistic_symbols.write(f"{entity}\n")
                    slot = partial_text["slot_name"].upper()
                    utt_semantic.append(slot)
                    non_linguistic_symbols.write(f"{slot}\n")
                    utt_semantic.append(partial_text["text"].lower())
            utt_text = utt_text.strip("\n")
            utt_text = utt_text.replace("\n", " ")
            utt_semantic = [
                utt_semantic[0]
            ]  # Currently, focus on intent classification task only
            utt_semantic = " ".join(utt_semantic)
            if (
                utt_text != "Turn the lights up"
            ):  # 1310.wav missed in the original dataset
                semantics.write(f"{utt_text}|{utt_semantic}\n")
    for wav in meta_info.values():
        wav, text, spk_id = wav["filename"], wav["text"], wav["worker"]["id"]
        wav_name = spk_id + "-" + wav
        text = text.strip("\n")
        text = text.replace("\n", " ")
        text_f.write(f"{text} {wav_name}\n")
        wav_path = args.wav_path + "/speech_corpus/audio/" + wav
        wavscp_f.write(f"{wav_name} {wav_path}\n")
        utt2spk_f.write(f"{wav_name} {spk_id}\n")

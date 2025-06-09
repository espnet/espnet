#!/usr/bin/env python3

# Copyright 2021 University of Stuttgart (Pavel Denisov)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import os
import re

import yaml

parser = argparse.ArgumentParser(
    description="Prepare IWSLT'21 Low Resource Speech Translation data."
)

parser.add_argument(
    "path", type=str, help="Path to IWSLT'21 Low Resource Speech Translation data"
)
parser.add_argument(
    "--raw-transcriptions",
    default=False,
    dest="raw",
    action="store_true",
    help="Do not convert transcriptions from the written to spoken language.",
)

args = parser.parse_args()

num_re = re.compile(r"(\d+\s*([\.,\/]\d+|)(%|))")
non_alpha_re = re.compile("[^a-z' <>]")

strings = {}


def num2words(matchobj):
    num = matchobj.group(0)
    if num in strings:
        return strings[num]
    else:
        return " <spn> "


valid_utts = []

with open("downloads/swahili-asr-resources/valid-iwslt.txt") as f:
    valid_utts = [line.split()[0] for line in f]

for pair in ["swa-eng", "swc-fra"]:
    sw = pair[:3]

    strings = {}

    with open("downloads/swahili-asr-resources/num2words-{}.txt".format(sw)) as f:
        for line in f:
            num, string = line.strip().split("\t")
            strings[num] = string

    subsets = {"train": [], "valid": [], "test": []}

    for subset in ["train", "valid"]:
        path = os.path.join(args.path, pair, subset)

        with open(os.path.join(path, "txt", subset + ".yaml")) as metafile, open(
            os.path.join(path, "txt", subset + ".swa")
        ) as textfile:
            wavs = yaml.safe_load(metafile)
            texts = textfile.readlines()

            for i in range(len(wavs)):
                if wavs[i]["wav"][-4:] != ".wav":
                    wavs[i]["wav"] += ".wav"

                uttid = "iwslt_" + wavs[i]["wav"][:-4].replace("/", "-")
                wav = os.path.join(path, "wav", wavs[i]["wav"])
                sentence = texts[i].strip()

                if subset == "train":
                    if uttid in valid_utts:
                        split = "valid"
                    else:
                        split = "train"
                else:
                    split = "test"

                if not args.raw:
                    sentence = re.sub(num_re, num2words, sentence)
                    sentence = sentence.lower()
                    sentence = re.sub(non_alpha_re, " ", sentence)

                sentence = " ".join(sentence.split())

                subsets[split].append({"wav": wav, "text": sentence, "id": uttid})

    for subset in subsets:
        odir = "data/{}_iwslt_{}".format(subset, sw)

        if args.raw:
            odir += "_raw"

        os.makedirs(odir, exist_ok=True)

        with open(odir + "/text", "w", encoding="utf-8") as text, open(
            odir + "/wav.scp", "w"
        ) as wavscp, open(odir + "/utt2spk", "w") as utt2spk:
            for utt in subsets[subset]:
                wavscp.write(
                    "{} sox --norm=-1 {}".format(utt["id"], utt["wav"])
                    + " -r 16k -t wav -c 1 -b 16 -e signed - |\n"
                )
                text.write("{} {}\n".format(utt["id"], utt["text"]))
                utt2spk.write("{} {}\n".format(utt["id"], utt["id"]))

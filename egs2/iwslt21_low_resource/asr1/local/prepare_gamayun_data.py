#!/usr/bin/env python3

# Copyright 2021 University of Stuttgart (Pavel Denisov)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import csv
import os
import re
import sys

num_re = re.compile(r"(\d+\s*([\.,\/]\d+|)(%|))")
non_alpha_re = re.compile("[^a-z' <>]")

valid_sentences = []

with open("downloads/swahili-asr-resources/valid-iwslt.txt") as f:
    for line in f:
        utt, text = line.split(" ", maxsplit=1)
        valid_sentences.append(text.strip())

strings = {}

with open("downloads/swahili-asr-resources/num2words-swa.txt") as f:
    for line in f:
        num, string = line.strip().split("\t")
        strings[num] = string


def num2words(matchobj):
    num = matchobj.group(0)
    if num in strings:
        return strings[num]
    else:
        return " <spn> "


idir = sys.argv[1]

odir = "data/train_gamayun"
os.makedirs(odir, exist_ok=True)

with open(odir + "/text", "w", encoding="utf-8") as text, open(
    odir + "/wav.scp", "w"
) as wavscp, open(odir + "/utt2spk", "w") as utt2spk, open(
    os.path.join(idir, "swahili_minikit.csv")
) as meta:
    reader = csv.reader(meta, delimiter="\t")
    for row in reader:
        uttid = "gamayun_" + row[0]
        wav = os.path.join(idir, "swahili_minikit", row[1])
        words = row[3]
        if words in valid_sentences:
            continue
        words = re.sub(num_re, num2words, words)
        words = words.lower()
        words = re.sub(non_alpha_re, " ", words)
        words = " ".join(words.split())
        wavscp.write(
            "{} sox --norm=-1 {} -r 16k -t wav -c 1 -b 16 -e signed - |\n".format(
                uttid, wav
            )
        )
        text.write("{} {}\n".format(uttid, words))
        utt2spk.write("{} {}\n".format(uttid, uttid))

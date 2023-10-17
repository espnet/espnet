#!/usr/bin/env python3

import os
import sys

from espnet2.text.phoneme_tokenizer import PhonemeTokenizer

idir = sys.argv[1]

tokenizer = PhonemeTokenizer("espeak_ng_english_us_vits")

with open(f"{idir}/text", encoding="utf-8") as itext, open(
    f"{idir}/text.phn", "w", encoding="utf-8"
) as otext:
    for line in itext:
        utt, text = line.strip("\n").split(" ", maxsplit=1)
        tokens = tokenizer.text2tokens(text)
        text_phn = "".join(tokens).replace("<space>", " ")
        otext.write(f"{utt} {text_phn}\n")

os.replace(f"{idir}/text", f"{idir}/text.orig")
os.replace(f"{idir}/text.phn", f"{idir}/text")

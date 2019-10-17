#!/usr/bin/env python3

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import codecs
import nltk
import re

from text.cleaners import english_cleaners

# https://github.com/Kyubyong/g2p
from g2p_en import G2p
f_g2p = G2p()


def g2p(text):
    tokens = filter(lambda s: s != " ", f_g2p(text))
    return ' '.join(tokens)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('text', type=str, help='text to be cleaned')
    parser.add_argument("trans_type", type=str, default="kana",
                        choices=["char", "phn"],
                        help="Input transcription type")
    args = parser.parse_args()
    with codecs.open(args.text, 'r', 'utf-8') as fid:
        for line in fid.readlines():
            id, _, content = line.split("|")
            clean_content = english_cleaners(content.rstrip())
            if args.trans_type == "phn":
                text = clean_content.lower()
                clean_content = g2p(text)

            print("%s %s" % (id, clean_content))

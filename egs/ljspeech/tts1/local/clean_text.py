#!/usr/bin/env python3

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import codecs
import nltk
import re

from text.cleaners import english_cleaners

_arpabet = nltk.corpus.cmudict.dict()

possible_symbols = '!\'(),-.:;?'


def _maybe_get_arpabet(word):
    assert " " not in word
    if len(word) > 1:
        # check that symbols are correctly tokenized in advance
        # e.g. if the word is 'smith.', this should raise
        if word[-1] in possible_symbols:
            raise RuntimeError("Oops! symbols should be correctly tokenized in advance")
    try:
        phonemes = _arpabet[word][0]
        phonemes = " ".join(phonemes)
    except KeyError:
        # return chars for failure cases
        return " ".join(list(word))

    return phonemes


def clean_manually(text):
    text = re.sub(r"'([\w ]+)',", r"\1,", text)
    # 'Edinburgh Review,'
    text = re.sub(r"'([\w ]+),'", r"\1,", text)
    return text


def g2p(text):
    words = nltk.word_tokenize(text)

    # Manual spacial handling...
    # remove last symbol if nltk fails to tokenize (e.g., Smith. -> Smith)
    words = map(lambda s: s[:-1] if len(s) > 1 and s[-1] in possible_symbols else s, words)
    # etc. -> etc
    words = map(lambda s: s.replace("etc.", "etc"), words)

    text = ' '.join(_maybe_get_arpabet(word) for word in words)
    return text


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
                text = clean_manually(text)
                clean_content = g2p(text)

            print("%s %s" % (id, clean_content))

#!/usr/bin/env python3

# Copyright 2019 Ryuichi Yamamoto
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import codecs

import pyopenjtalk
import jaconv


def g2p(text, input_type="kana"):
    text = jaconv.normalize(text)
    if input_type == "kana":
        text = pyopenjtalk.g2p(text, kana=True)
        text = jaconv.kata2hira(text)
    elif input_type == "hira":
        text = pyopenjtalk.g2p(text, kana=True)
    elif input_type == "phoneme":
        text = pyopenjtalk.g2p(text, kana=False)
    else:
        assert False
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('text', type=str, help='text to be cleaned')
    parser.add_argument("input_type", type=str, default="kana",
                        choices=["kana", "hira", "phoneme"],
                        help="Input text type")
    args = parser.parse_args()
    with codecs.open(args.text, 'r', 'utf-8') as fid:
        for line in fid.readlines():
            id, content = line.split(":")
            clean_content = g2p(content, args.input_type)
            print("%s %s" % (id, clean_content))

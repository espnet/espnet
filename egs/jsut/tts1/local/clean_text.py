#!/usr/bin/env python3

# Copyright 2018 Nagoya University (Takenori Yoshimura), Ryuichi Yamamoto
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import codecs

import pyopenjtalk
import jaconv


def g2p(text, trans_type="char"):
    text = jaconv.normalize(text)
    if trans_type == "char":
        text = pyopenjtalk.g2p(text, kana=True)
    elif trans_type == "phn":
        text = pyopenjtalk.g2p(text, kana=False)
    else:
        assert False
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
            id, content = line.split(":")
            clean_content = g2p(content, args.trans_type)
            print("%s %s" % (id, clean_content))

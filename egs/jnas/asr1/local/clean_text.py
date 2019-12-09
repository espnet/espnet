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
    parser.add_argument('in_text', type=str, help='text to be cleaned')
    parser.add_argument('out_text', type=str, help='text to be cleaned')
    parser.add_argument("trans_type", type=str, default="kana",
                        choices=["char", "phn"],
                        help="Input transcription type")
    args = parser.parse_args()
    with codecs.open(args.in_text, 'r', 'utf-8') as f_in, codecs.open(args.out_text, 'w', 'utf-8') as f_out:
        for line in f_in.readlines():
            id = line.split(" ")[0]
            content = "".join(line.split(" ")[1:])
            clean_content = g2p(content, args.trans_type)
            f_out.write("%s %s\n" % (id, clean_content))

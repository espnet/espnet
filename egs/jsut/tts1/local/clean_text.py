#!/usr/bin/env python

# Copyright 2018 Nagoya University (Takenori Yoshimura)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import codecs

from text.text2yomi import Text2Yomi


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dict', type=str, help='dictionary to be used')
    parser.add_argument('text', type=str, help='text to be cleaned')
    args = parser.parse_args()
    text2yomi = Text2Yomi(args.dict)
    with codecs.open(args.text, 'r', 'utf-8') as fid:
        for line in fid.readlines():
            id, content = line.split(":")
            clean_content = text2yomi(content)
            print("%s %s" % (id, clean_content))

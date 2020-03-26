#!/usr/bin/env python3

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import codecs

from text.cleaners import custom_english_cleaners


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('text', type=str, help='text to be cleaned')
    args = parser.parse_args()
    with codecs.open(args.text, 'r', 'utf-8') as fid:
        for line in fid.readlines():
            line = line.split(" ")
            id = line[0]
            content = " ".join(line[1:])
            clean_content = custom_english_cleaners(content.rstrip())
            print("%s %s" % (id, clean_content))

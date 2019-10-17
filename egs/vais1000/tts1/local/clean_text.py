#!/usr/bin/env python3

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# edited by enamoria: add vietnamese cleaner

import argparse
import codecs
import os
import pdb

from vietnamese_cleaners import vietnamese_naive_cleaner


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('text', type=str, help='text to be cleaned')
    args = parser.parse_args()

    lines = {}
    with codecs.open(args.text, 'r', 'utf-8') as fid:
        for line in fid.readlines():
            id, _, content = line.split("|")
            
            clean_content = vietnamese_naive_cleaner(content)
            lines[id] = clean_content

        # TODO This below hack is for sorting the text. Since python sorting and the bash sorting is different (is it???), so I write the text to a temporary text file, then sort it using bash. This is ugly I know, but I'm not quite familiar with kaldi and their stuff, so I simply try to fit their pipeline

        with codecs.open("text_temp", "w", "utf8") as f:
            for id in sorted(lines.keys()):
                f.write(f"{id} {lines[id]}\n")
        
        os.system("awk '{print $1}' text_temp | sort -o text_temp")

        with codecs.open("text_temp", "r", "utf8") as f:
            for line in f.readlines():
                if line and line.strip():
                    print(f"{line.strip()} {lines[line.strip()]}")

                    # print(lines[line])
                    # print(lines)
                    # break


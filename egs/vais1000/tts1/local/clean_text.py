#!/usr/bin/env python3

# Copyright 2018 Nagoya University (Tomoki Hayashi) and Kính Phan (@enamoria)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import codecs

from vietnamese_cleaner.vietnamese_cleaners import vietnamese_cleaner

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("text", type=str, help="text to be cleaned")
    args = parser.parse_args()

    lines = {}
    with codecs.open(args.text, "r", "utf-8") as fid:
        for line in fid.readlines():
            id, _, content = line.split("|")

            clean_content = vietnamese_cleaner(content)
            lines[id] = clean_content

        for id in sorted(lines.keys()):
            print(f"{id} {lines[id]}")

#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2021 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import codecs

parser = argparse.ArgumentParser()
parser.add_argument("--utt2spk", type=str, help="path to utt2spk file")
parser.add_argument("--offlimit_list", type=str, help="path to offllimit list file")
args = parser.parse_args()


def main():
    offlimit_spk_list = []
    with codecs.open(args.offlimit_list, "r", encoding="utf-8") as f:
        for line in f:
            offlimit_spk_list.append(int(line.strip()))

    with codecs.open(args.utt2spk, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            spk = line.split(" ")[1]

            if "ted_" in spk:
                spk_id = int(spk.split("_")[-1])
                if spk_id in offlimit_spk_list:
                    continue
            print(line)


if __name__ == "__main__":
    main()

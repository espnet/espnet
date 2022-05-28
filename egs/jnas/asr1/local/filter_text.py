#!/usr/bin/env python3

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


import argparse
import codecs
import sys
from io import open

PY2 = sys.version_info[0] == 2
sys.stdin = codecs.getreader("utf-8")(sys.stdin if PY2 else sys.stdin.buffer)
sys.stdout = codecs.getwriter("utf-8")(sys.stdout if PY2 else sys.stdout.buffer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter-list", "-f", type=str, help="filter list")
    args = parser.parse_args()

    with open(args.filter_list, encoding="utf-8") as f:
        fil = [x.rstrip() for x in f]

    for x in sys.stdin:
        # extract text parts
        text = " ".join(x.rstrip().split()[1:])
        if text in fil:
            print(x.split()[0], text)

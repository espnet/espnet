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
    parser.add_argument(
        "--skip-ncols", "-s", default=0, type=int, help="skip first n columns"
    )
    parser.add_argument("text", type=str, help="input text")
    args = parser.parse_args()

    if args.text:
        f = open(args.text, encoding="utf-8")
    else:
        f = sys.stdin

    for line in f:
        x = line.split()
        print(" ".join(x[: args.skip_ncols]), end=" ")
        print(" ".join([st.split("+")[0] for st in x[args.skip_ncols :]]))

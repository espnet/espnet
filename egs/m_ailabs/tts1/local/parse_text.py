#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# Copyright 2019 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import json
import sys


IS_PY2 = sys.version_info[0] == 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsons", nargs="+", type=str,
                        help="*_mls.json filenames")
    parser.add_argument("--spk", type=str,
                        help="speaker tag")
    args = parser.parse_args()

    for filename in args.jsons:
        with open(filename, "r") as f:
            js = json.load(f)
        for key in js.keys():
            uid = args.spk + "_" + key[:-4]
            text = js[key]["clean"].upper()
            if IS_PY2:
                print(u"%s %s".encode("utf-8") % (uid, text))
            else:
                print("%s %s" % (uid, text))


if __name__ == "__main__":
    main()

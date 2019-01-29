#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# Copyright 2019 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import codecs
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsons", nargs="+", type=str,
                        help="*_mls.json filenames")
    parser.add_argument("--spk", type=str,
                        help="speaker tag")
    parser.add_argument("out", type=str,
                        help="output filename")
    args = parser.parse_args()

    dirname = os.path.dirname(args.out)
    if len(dirname) != 0 and not os.path.exists(dirname):
        os.makedirs(dirname)

    with codecs.open(args.out, "w", encoding="utf-8") as out:
        for filename in sorted(args.jsons):
            with codecs.open(filename, "r", encoding="utf-8") as f:
                js = json.load(f)
            for key in sorted(js.keys()):
                uid = args.spk + "_" + key[:-4]
                text = js[key]["clean"].upper()
                out.write("%s %s\n" % (uid, text))


if __name__ == "__main__":
    main()

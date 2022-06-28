#!/usr/bin/env python

# Copyright 2021 Takenori Yoshimura
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import re

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-header", action="store_true", help="if true, skip first line"
    )
    parser.add_argument("text", type=str, help="text to be cleaned")
    args = parser.parse_args()
    with open(args.text, "r", encoding="utf-8") as fid:
        if args.skip_header:
            fid.readline()
        for line in fid.readlines():
            id, content = line.split(",")
            content = re.sub("（.*?）", "", content.rstrip())
            content = re.sub("「(.*?)」", "\\1", content)
            print("%s %s" % (id, content))

#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#           2018 Xuankai Chang (Shanghai Jiao Tong University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import codecs
import json
import re
import sys

is_python2 = sys.version_info[0] == 2


def get_parser():
    parser = argparse.ArgumentParser(
        description="convert sclite's result.txt file to json"
    )
    parser.add_argument("--key", "-k", type=str, help="key")
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    key = re.findall(r"r\d+h\d+", args.key)[0]

    re_id = r"^id: "
    re_strings = {
        "Speaker": r"^Speaker sentences",
        "Scores": r"^Scores: ",
        "REF": r"^REF: ",
        "HYP": r"^HYP: ",
    }
    re_id = re.compile(re_id)
    re_patterns = {}
    for p in re_strings.keys():
        re_patterns[p] = re.compile(re_strings[p])

    ret = {}
    tmp_id = None
    tmp_ret = {}

    sys.stdin = codecs.getreader("utf-8")(sys.stdin if is_python2 else sys.stdin.buffer)
    sys.stdout = codecs.getwriter("utf-8")(
        sys.stdout if is_python2 else sys.stdout.buffer
    )
    line = sys.stdin.readline()
    while line:
        x = line.rstrip()
        x_split = x.split()

        if re_id.match(x):
            if tmp_id:
                ret[tmp_id] = {key: tmp_ret}
                tmp_ret = {}
            tmp_id = x_split[1]
        for p in re_patterns.keys():
            if re_patterns[p].match(x):
                tmp_ret[p] = " ".join(x_split[1:])
        line = sys.stdin.readline()

    if tmp_ret != {}:
        ret[tmp_id] = {key: tmp_ret}

    all_l = {"utts": ret}
    # ensure "ensure_ascii=False", which is a bug
    jsonstring = json.dumps(
        all_l, indent=4, ensure_ascii=False, sort_keys=True, separators=(",", ": ")
    )
    print(jsonstring)

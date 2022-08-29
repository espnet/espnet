#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2020 Nagoya University (Wen-Chin Huang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import json
import logging
import sys
from io import open

from espnet.utils.cli_utils import get_commandline_args


def get_parser():
    parser = argparse.ArgumentParser(
        description="Merge source and target data.json files into one json file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--src-json", type=str, help="Json file for the source speaker")
    parser.add_argument(
        "--trg-json",
        type=str,
        default=None,
        help="Json file for the target speaker. If not specified, use source only.",
    )
    parser.add_argument(
        "--num_utts", default=-1, type=int, help="Number of utterances (take from head)"
    )
    parser.add_argument("--verbose", "-V", default=1, type=int, help="Verbose option")
    parser.add_argument(
        "--out",
        "-O",
        type=str,
        help="The output filename. " "If omitted, then output to sys.stdout",
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    # logging info
    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    if args.verbose > 0:
        logging.basicConfig(level=logging.INFO, format=logfmt)
    else:
        logging.basicConfig(level=logging.WARN, format=logfmt)
    logging.info(get_commandline_args())

    with open(args.src_json, "rb") as f:
        src_json = json.load(f)["utts"]
    if args.trg_json:
        with open(args.trg_json, "rb") as f:
            trg_json = json.load(f)["utts"]

    # get source and target speaker
    _ = list(src_json.keys())[0].split("_")
    srcspk = _[0]
    if args.trg_json:
        _ = list(trg_json.keys())[0].split("_")
        trgspk = _[0]

    count = 0
    data = {"utts": {}}
    # (dirty) loop through input only because in/out should have same files
    for k, v in src_json.items():
        _ = k.split("_")
        number = "_".join(_[1:])

        entry = {"input": src_json[srcspk + "_" + number]["input"]}

        if args.trg_json:
            entry["output"] = trg_json[trgspk + "_" + number]["input"]
            entry["output"][0]["name"] = "target1"

        data["utts"][number] = entry
        count += 1
        if args.num_utts > 0 and count >= args.num_utts:
            break

    if args.out is None:
        out = sys.stdout
    else:
        out = open(args.out, "w", encoding="utf-8")

    json.dump(
        data, out, indent=4, ensure_ascii=False, separators=(",", ": "),
    )

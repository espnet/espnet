#!/usr/bin/env python3

import argparse
import codecs
import math
import re
import sys

is_python2 = sys.version_info[0] == 2


def get_parser():
    parser = argparse.ArgumentParser(description="convert trn to ctm")
    parser.add_argument("trn", type=str, default=None, nargs="?", help="input trn")
    parser.add_argument("ctm", type=str, default=None, nargs="?", help="output ctm")
    return parser


def main(args):
    args = get_parser().parse_args(args)
    convert(args.trn, args.ctm)


def convert(trn=None, ctm=None):
    if trn is not None:
        with codecs.open(trn, "r", encoding="utf-8") as trn:
            content = trn.readlines()
    else:
        trn = codecs.getreader("utf-8")(sys.stdin if is_python2 else sys.stdin.buffer)
        content = trn.readlines()
    split_content = []
    for i, line in enumerate(content):
        idx = line.rindex("(")
        split = [line[:idx].strip().upper(), line[idx + 1 :].strip()[:-1]]
        while "((" in split[0]:
            split[0] = split[0].replace("((", "(")
        while "  " in split[0]:
            split[0] = split[0].replace("  ", " ")
        segm_info = re.split("[-_]", split[1])
        segm_info = [s.strip() for s in segm_info]
        col1 = segm_info[0] + "_" + segm_info[1]
        col2 = segm_info[2]
        start_time_int = int(segm_info[6])
        end_time_int = int(segm_info[7])
        diff_int = end_time_int - start_time_int
        word_split = split[0].split(" ")
        word_split = list(
            filter(lambda x: len(x) > 0 and any([c != " " for c in x]), word_split)
        )
        if len(word_split) > 0:
            step_int = int(math.floor(float(diff_int) / len(word_split)))
            step = str(step_int)
            for j, word in enumerate(word_split):
                start_time = str(int(start_time_int + step_int * j))
                col3 = (
                    (start_time[:-2] if len(start_time) > 2 else "0")
                    + "."
                    + (start_time[-2:] if len(start_time) > 1 else "00")
                )
                if j == len(word_split) - 1:
                    diff = str(int(end_time_int - int(start_time)))
                else:
                    diff = step
                col4 = (diff[:-2] if len(diff) > 2 else "0") + "." + diff[-2:]
                segm_info = [col1, col2, col3, col4]
                split_content.append(" ".join(segm_info) + "  " + word)
    if ctm is not None:
        sys.stdout = codecs.open(ctm, "w", encoding="utf-8")
    else:
        sys.stdout = codecs.getwriter("utf-8")(
            sys.stdout if is_python2 else sys.stdout.buffer
        )
    for c_line in split_content:
        print(c_line)


if __name__ == "__main__":
    main(sys.argv[1:])

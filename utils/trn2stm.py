#!/usr/bin/python

import argparse
import codecs
import re
import sys

is_python2 = sys.version_info[0] == 2


def get_parser():
    parser = argparse.ArgumentParser(description='convert trn to stm')
    parser.add_argument('--orig-stm', type=str, default=None, nargs='?',
                        help="Original stm file to add additional information to the generated one")
    parser.add_argument('trn', type=str, default=None, nargs='?',
                        help='input trn')
    parser.add_argument('stm', type=str, default=None, nargs='?', help='output stm')
    return parser


def main(args):
    args = get_parser().parse_args(args)
    convert(args.trn, args.stm, args.orig_stm)


def convert(trn=None, stm=None, orig_stm=None):
    if orig_stm is not None:
        with codecs.open(orig_stm, 'r', encoding="utf-8") as orig_stm:
            orig_content = orig_stm.readlines()
            has_orig = True
            header = []
            content = []
            for line in orig_content:
                (header if line.startswith(";;") else content).append(line.strip())
            del orig_content
            content = [x.split(" ") for x in content]
            mapping = {}
            for x in content:
                mapping[x[2]] = x[5]
            del content
    else:
        has_orig = False
        header = None
        mapping = None

    if trn is not None:
        with codecs.open(trn, 'r', encoding="utf-8") as trn:
            content = trn.readlines()
    else:
        trn = codecs.getreader("utf-8")(sys.stdin if is_python2 else sys.stdin.buffer)
        content = trn.readlines()

    for i, line in enumerate(content):
        idx = line.rindex("(")
        split = [line[:idx].strip().upper() + " ", line[idx + 1:].strip()[:-1]]
        while "((" in split[0]:
            split[0] = split[0].replace("((", "(")
        while "  " in split[0]:
            split[0] = split[0].replace("  ", " ")
        segm_info = re.split("[-_]", split[1])
        segm_info = [s.strip() for s in segm_info]
        col1 = segm_info[0] + "_" + segm_info[1]
        col2 = segm_info[2]
        col3 = segm_info[3] + "_" + segm_info[4] + "_" + segm_info[5]
        start_time = str(int(segm_info[6]))
        end_time = str(int(segm_info[7]))
        col4 = (start_time[:-2] if len(start_time) > 2 else "0") + "." + (
            start_time[-2:] if len(start_time) > 1 else "00")
        col5 = (end_time[:-2] if len(end_time) > 2 else "0") + "." + (end_time[-2:] if len(end_time) > 1 else "00")
        col6 = mapping[col3] if has_orig else ""
        segm_info = [col1, col2, col3, col4, col5, col6]
        content[i] = " ".join(segm_info) + "  " + split[0]
    if stm is not None:
        sys.stdout = codecs.open(stm, "w", encoding="utf-8")
    else:
        sys.stdout = codecs.getwriter("utf-8")(
            sys.stdout if is_python2 else sys.stdout.buffer)
    if has_orig:
        for h_line in header:
            print(h_line)
    for c_line in content:
        print(c_line)


if __name__ == "__main__":
    main(sys.argv[1:])

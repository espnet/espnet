#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2021 Johns Hopkins University (Jiatong Shi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import codecs
import sys


def get_parser():
    parser = argparse.ArgumentParser(description="language identification scoring")
    parser.add_argument("--ref", type=str, help="input reference", required=True)
    parser.add_argument("--hyp", type=str, help="input hypotheses", required=True)
    parser.add_argument(
        "--out",
        type=argparse.FileType("w"),
        default=sys.stdout,
        help="The output filename. " "If omitted, then output to sys.stdout",
    )
    return parser


def main(args):
    args = get_parser().parse_args(args)
    scoring(args.ref, args.hyp, args.out)


def scoring(ref, hyp, out):
    ref_file = codecs.open(ref, "r", encoding="utf-8")
    hyp_file = codecs.open(hyp, "r", encoding="utf-8")

    utt_num = 0
    correct = 0

    while True:
        ref_utt = ref_file.readline()
        hyp_utt = hyp_file.readline()

        if not ref_utt or not hyp_utt:
            break

        [rec_id, lid, utt_id] = ref_utt.strip().split()
        [hrec_id, hlid, hutt_id] = hyp_utt.strip().split()

        assert (rec_id == hrec_id and utt_id == hutt_id) and "Mismatch in trn id"

        if lid == hlid:
            correct += 1
        utt_num += 1
    out.write(
        "Language Identification Scoring: Accuracy {:.4f} ({}/{})".format(
            (correct / float(utt_num)), correct, utt_num
        )
    )


if __name__ == "__main__":
    main(sys.argv[1:])

#!/usr/bin/env python3
import argparse
import math
import os
import sys

from espnet2.fileio.read_text import read_label
from espnet2.fileio.score_scp import SingingScoreReader, SingingScoreWriter
from espnet2.text.build_tokenizer import build_tokenizer

"""Check alignment between label and score at phone level."""


def load_customed_dic(file):
    """If syllable-to-phone tranlation differs from g2p,"""
    """ customed tranlation can be added to customed_dic."""
    customed_dic = {}
    with open(file, "r", encoding="utf-8") as f:
        content = f.read().strip().split("\n")
        for key in content:
            key = key.split(" ")
            customed_dic[key[0]] = key[1].split("_")
    return customed_dic


def compare(key, score, label):
    customed_dic = load_customed_dic(args.customed_dic)
    tokenizer = build_tokenizer(
        token_type="phn",
        bpemodel=None,
        delimiter=None,
        space_symbol="<space>",
        non_linguistic_symbols=None,
        g2p_type=args.g2p,
    )
    index = 0
    score_error = -1
    phoneme_error = -1
    for i in range(len(score)):
        syb = score[i][2]
        # Translate syllable into phones through g2p
        phns = tokenizer.g2p(syb)
        # In some case, translation can be different
        if syb in customed_dic:
            phns = customed_dic[syb]
        score[i].append("_".join(phns))
        for p in phns:
            if index >= len(labels):
                raise ValueError("Syllables are longer than phones in {}".format(key))
            elif label[index][2] == p:
                index += 1
            else:
                raise ValueError(
                    "Mismatch in syllable [{}]->{} and {}-th phones '{}' in {}.".format(
                        syb, phns, index, label[index][2], key
                    )
                )
    if index != len(label):
        raise ValueError("Syllables are shorter than phones in {}: ".format(key))
    return score


def get_parser():
    parser = argparse.ArgumentParser(
        description="Compare segments between label and score",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("scp", type=str, help="scp folder")
    parser.add_argument("--g2p", type=str, help="g2p", default="pyopenjtalk")
    parser.add_argument(
        "--score_dump", type=str, default="score_dump", help="score dump directory"
    )
    parser.add_argument(
        "--customed_dic",
        type=str,
        help="customed g2p for alignment at phoneme level",
        default="local/customed_dic.scp",
    )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    reader = SingingScoreReader(os.path.join(args.scp, "score.scp"))
    writer = SingingScoreWriter(
        args.score_dump, os.path.join(args.scp, "score.scp.tmp")
    )
    labels = read_label(os.path.join(args.scp, "label"))
    for key in labels:
        score = reader[key]
        score["note"] = compare(key, score["note"], labels[key])
        score["item_list"].append("phn")
        writer[key] = score

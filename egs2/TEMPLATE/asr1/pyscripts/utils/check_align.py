#!/usr/bin/env python3
import argparse
import math
import os
import re
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
    for i in range(len(score)):
        syb = score[i][2]
        # multi note in one syllable
        if syb == "—":
            if index < len(label):
                if label[index][2] == pre_phn:
                    score[i].append(pre_phn)
                    index += 1
                    continue
                else:
                    raise ValueError(
                        "Mismatch of slur in [{}]->{} and {}-th '{}' in {}.".format(
                            syb, pre_phn, index, label[index][2], key
                        )
                    )
            else:
                raise ValueError("Syllables are longer than phones in {}".format(key))

        # Translate syllable into phones through g2p
        phns = tokenizer.g2p(syb)
        # In some case, translation can be different
        if syb in customed_dic:
            phns = customed_dic[syb]
        score[i].append("_".join(phns))
        pre_phn = phns[-1]
        for p in phns:
            if index >= len(label):
                pattern = r"_[0]*"
                key_name = re.split(pattern, key[:-5], 1)[-1]
                print(
                    "Error in {}, copy this code to `get_error_dict`"
                    ' in prep_segments.py under `if input_type == "xml"`.\n'
                    '"{}": [\n'
                    "   lambda i, labels, segment, segments, threshold: "
                    "add_pause(labels, segment, segments, threshold)\n"
                    '   if (labels[i].lyric == "{}" and labels[i - 1].lyric == "{}")\n'
                    "   else (labels, segment, segments, False),\n"
                    "],".format(key, key_name, score[i][2], score[i - 1][2])
                )
                raise ValueError("Lyrics are longer than phones in {}".format(key))
            elif label[index][2] == p:
                index += 1
            else:
                raise ValueError(
                    "Mismatch in syllable [{}]->{} and {}-th phones '{}' in {}.".format(
                        syb, phns, index, label[index][2], key
                    )
                )
    if index != len(label):
        pattern = r"_[0]*"
        key_name = re.split(pattern, key[:-5], 1)[-1]
        print(
            "Error in {}, copy this code to `get_error_dict` in prep_segments.py"
            ' under `if input_type == "hts"`.\n'
            '"{}": [\n'
            "   lambda i, labels, segment, segments, threshold: "
            "add_pause(labels, segment, segments, threshold)\n"
            '   if (labels[i].label_id == "{}" and labels[i - 1].label_id == "{}")\n'
            "   else (labels, segment, segments, False),\n"
            "],".format(key, key_name, label[index][2], label[index - 1][2])
        )
        raise ValueError("Phones are longer than lyrics in {}.".format(key))
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

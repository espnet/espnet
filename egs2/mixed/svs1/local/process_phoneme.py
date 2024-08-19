#!/usr/bin/env python3
import os
import re
import sys
import argparse

from espnet2.fileio.read_text import read_label
from espnet2.fileio.score_scp import SingingScoreReader, SingingScoreWriter
from ACE_phonemes.main import jp_word_to_phoneme, pinyin_to_phoneme

"""Process origin phoneme in music score using ACE-phoneme"""

jp_datasets = [
    "ameboshi"
]

zh_datasets = [
    "opencpop"
]


def check_language(dataset):
    if dataset in jp_datasets:
        return "jp"
    elif dataset in zh_datasets:
        return "zh"
    else:
        raise ValueError(f"{dataset} is not supported.")


def convert(
    key, 
    score, 
    labels, 
    phn_seg = {
        1: [1],
        2: [0.25, 1],
        3: [0.1, 0.5, 1],
        4: [0.05, 0.1, 0.5, 1],
    },
):
    index = 0
    new_labels = []
    dataset = key.split("_")[0]
    for i in range(len(score)):
        lyric = score[i][2]
        org_phns = score[i][4]
        if check_language(dataset) == "jp":
            ace_phns = jp_word_to_phoneme(lyric)
        else:
            ace_phns = pinyin_to_phoneme(lyric)
        if len(org_phns) == len(ace_phns):
            for i in range(len(org_phns)):
                if org_phns[i] != ace_phns[i]:
                    print("Warning: Mismatch in syllable [{}]-> ace: {} and org: {}, {}-th phoneme {} vs {} in {}".format(
                            lyric, ace_phns, org_phns, i, ace_phns[i], org_phns[i], key
                    ))
                new_labels.append([labels[index][0], labels[index][1], ace_phns[i]])
                index += 1
        else:
            print("Error: Different length in syllable [{}]-> ace: {} and org: {}".format(ace_phns, org_phns))
            tot_dur = 0
            for i in range(len(org_phns)):
                tot_dur += labels[index + i][2]
            print(tot_dur)
            pre_seg = 0
            for i in range(len(ace_phns)):
                phn_ruled_dur = (phn_seg[i] - pre_seg) * tot_dur
                pre_seg = pre_seg[i]
                print(f"{i}: {phn_ruled_dur}")
    return score


def get_parser():
    parser = argparse.ArgumentParser(
        description="Process origin phoneme in music score using ACE-phoneme",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scp", type=str, help="data directory scp")
    parser.add_argument(
        "--score_dump", default="score_dump", type=str, help="score dump directory"
    )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    reader = SingingScoreReader(os.path.join(args.scp, "score.scp"))
    writer = SingingScoreReader(
        args.score_dump,
        os.path.join(args.scp, "score.scp.tmp")
    )
    labels = read_label(os.path.join(args.scp, "label"))
    lablel_writer = open(os.path.join(args.scp, "label.tmp"))
    for key in labels:
        score = reader[key]
        score["note"], new_labels = convert(key, score["note"], labels[key])
        # writer[key] = score
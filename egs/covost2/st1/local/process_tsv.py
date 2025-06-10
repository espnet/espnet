#!/usr/bin/env python3

# Copyright 2021 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import codecs

import pandas as pd

parser = argparse.ArgumentParser(description="extract translation from tsv file")
parser.add_argument("tsv_path", type=str, default=None, help="input tsv path")
parser.add_argument("text", type=str, default=None, help="text path")
parser.add_argument("utt2spk", type=str, default=None, help="utt2spk path")
parser.add_argument(
    "save_path_src",
    type=str,
    default=None,
    help="output filtered transcription path in the source language",
)
parser.add_argument(
    "save_path_tgt",
    type=str,
    default=None,
    help="output translation path in the target language",
)
parser.add_argument("set", type=str, default=None, help="data split")
args = parser.parse_args()


def main():
    df = pd.read_csv(args.tsv_path, encoding="utf-8", delimiter="\t")
    df = df.loc[:, ["path", "translation", "split"]]

    if args.set == "train":
        df = df[(df["split"] == args.set) | (df["split"] == "train_covost")]
    else:
        df = df[df["split"] == args.set]
    # NOTE: following get_v2_split() in
    # https://github.com/facebookresearch/covost/blob/master/get_covost_splits.py
    data = df.to_dict(orient="index").items()
    data = [v for k, v in sorted(data, key=lambda x: x[0])]

    # read utt2spk (used to get complete speaker id missed in mp3 file name)
    mp3path2uttid = {}
    with codecs.open(args.utt2spk, "r", encoding="utf-8") as f:
        for line in f:
            utt_id, spk_id = line.strip().split(" ")
            mp3_name = "-".join(utt_id.split("-")[1:])
            mp3path2uttid[mp3_name] = utt_id
            # NOTE: utt_id = spk_id - mp3_name

    # filter transcription
    uttid2transcription = {}
    with codecs.open(args.text, "r", encoding="utf-8") as f:
        for line in f:
            utt_id = line.strip().split(" ")[0]
            transcription = " ".join(line.strip().split(" ")[1:])
            uttid2transcription[utt_id] = transcription

    # save translation
    with codecs.open(args.save_path_src, "w", encoding="utf-8") as f_src, codecs.open(
        args.save_path_tgt, "w", encoding="utf-8"
    ) as f_tgt:
        for d in data:
            if d["path"].split(".")[0] not in mp3path2uttid:
                print("Skip %s (empty mp3)" % d["path"])
                continue
            utt_id = mp3path2uttid[d["path"].split(".")[0]]
            if isinstance(d["translation"], float):
                print("Skip %s (empty translation)" % utt_id)
            else:
                f_tgt.write(utt_id + " " + d["translation"] + "\n")
                f_src.write(utt_id + " " + uttid2transcription[utt_id] + "\n")


if __name__ == "__main__":
    main()

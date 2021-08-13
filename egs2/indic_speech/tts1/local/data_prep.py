#!/usr/bin/env python3

# Copyright 2021 Carnegie Mellon University (Peter Wu)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


import argparse
import csv
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", help="downloads directory", type=str, default="downloads")
    args = parser.parse_args()

    spk = "Hindi_TTS_dataset"
    wav_dir = os.path.join(args.d, "%s/Dataset" % spk)
    annotations_path = os.path.join(args.d, "%s/annotations.csv" % spk)
    utt2text = {}
    utt2f = {}
    text_strs = []
    wav_scp_strs = []

    with open(annotations_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter="|")
        for row in csv_reader:
            f = row[0]  # e.g. Dataset/3487.wav
            f = os.path.basename(row[0])
            utt = f[:-4]
            utt = spk + "_" + utt
            utt2f[utt] = f
            text = row[1]
            utt2text[utt] = text

    sr = 16000
    utts = [utt for utt in utt2text]
    utts = sorted(utts)
    utts_str = " ".join(utts)
    spk2utt_str = "%s %s" % (spk, utts_str)
    text_strs = ["%s %s" % (utt, utt2text[utt]) for utt in utts]
    wav_scp_strs = []
    for utt in utts:
        cmd = "ffmpeg -i %s/%s -f wav -ar %d -ab 16 -ac 1 - |" % (
            wav_dir,
            utt2f[utt],
            sr,
        )
        wav_scp_strs.append("%s %s" % (utt, cmd))

    data_subdir = "data/%s" % spk
    if not os.path.exists(data_subdir):
        os.makedirs(data_subdir)
    with open(os.path.join(data_subdir, "text"), "w+") as ouf:
        for s in text_strs:
            ouf.write("%s\n" % s)
    with open(os.path.join(data_subdir, "wav.scp"), "w+") as ouf:
        for s in wav_scp_strs:
            ouf.write("%s\n" % s)
    with open(os.path.join(data_subdir, "spk2utt"), "w+") as ouf:
        ouf.write("%s\n" % spk2utt_str)

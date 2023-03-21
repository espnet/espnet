#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2021 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Merge adjacent utterances."""


import argparse
import codecs
from collections import deque

parser = argparse.ArgumentParser()
parser.add_argument("segments", type=str, help="path to segment file")

parser.add_argument("output_segments", type=str, help="path to output segment file")
parser.add_argument("output_utt2spk", type=str, help="path to output utt2spk file")
parser.add_argument("output_spk2utt", type=str, help="path to output spk2utt file")

parser.add_argument("--min_interval", type=int, default=200, help="")
parser.add_argument(
    "--max_duration", type=int, default=1500, help="maximum duration [frame]"
)
parser.add_argument(
    "--delimiter", type=str, default="_", help="delimiter on utt_id start_time end_time"
)
args = parser.parse_args()


def merge(segments, segments_dict):
    while True:
        num_merge = 0
        new_segments = deque([])
        utt_id_prev, start_prev, end_prev = segments.popleft()
        utt_ids_merged = utt_id_prev
        for utt_ids, start, end in segments:
            interval = start - end_prev
            duration = end - start_prev
            if interval < args.min_interval and duration < args.max_duration:
                # merge
                end_prev = end
                utt_ids_merged.extend(utt_ids)
                num_merge += 1
            else:
                new_segments.append((utt_ids_merged, start_prev, end_prev))

                start_prev = start
                end_prev = end
                utt_ids_merged = utt_ids

        # for last segments
        new_segments.append((utt_ids_merged, start_prev, end))
        segments = new_segments

        if num_merge == 0:
            break

    delimiter = args.delimiter
    for utt_ids, _, _ in segments:
        spk = delimiter.join(utt_ids[0].split(delimiter)[:-2])
        s = utt_ids[0].split(delimiter)[-2]
        e = utt_ids[-1].split(delimiter)[-1]
        new_utt_id = "%s" % (spk + delimiter + s + delimiter + e)

        segments_dict[new_utt_id] = (
            segments_dict[utt_ids[0]][0],
            segments_dict[utt_ids[-1]][1],
        )

        if len(utt_ids) > 1:
            for utt_id in utt_ids:
                del segments_dict[utt_id]

    return segments_dict


def main():
    segments_dict = {}
    with codecs.open(args.segments, "r", encoding="utf-8") as f:
        for line in f:
            utt_id, spk, start, end = line.strip().split()
            segments_dict[utt_id] = (start, end)

    segments_spk = deque([])
    with codecs.open(args.segments, "r", encoding="utf-8") as f:
        spk_prev = None
        for line in f:
            utt_id, spk, start, end = line.strip().split()
            start = float(start) * 100  # per 10ms
            end = float(end) * 100  # per 10ms
            if spk_prev is not None and spk != spk_prev:
                segments_dict = merge(segments_spk, segments_dict)
                segments_spk = deque([])  # reset
            segments_spk.append(([utt_id], start, end))
            spk_prev = spk

    with codecs.open(args.output_segments, "w", encoding="utf-8") as f:
        for utt_id, (start, end) in sorted(segments_dict.items(), key=lambda x: x[0]):
            spk = args.delimiter.join(utt_id.split(args.delimiter)[:-2])
            f.write("%s %s %s %s\n" % (utt_id, spk, start, end))

    spk2utt_dict = {}
    with codecs.open(args.output_utt2spk, "w", encoding="utf-8") as f:
        for utt_id, ref in sorted(segments_dict.items(), key=lambda x: x[0]):
            spk = args.delimiter.join(utt_id.split(args.delimiter)[:-2])
            f.write("%s %s\n" % (utt_id, spk))

            if spk not in spk2utt_dict:
                spk2utt_dict[spk] = [utt_id]
            else:
                spk2utt_dict[spk] += [utt_id]

    with codecs.open(args.output_spk2utt, "w", encoding="utf-8") as f:
        for spk, utt_ids in sorted(spk2utt_dict.items(), key=lambda x: x[0]):
            f.write("%s %s\n" % (spk, " ".join(utt_ids)))


if __name__ == "__main__":
    main()

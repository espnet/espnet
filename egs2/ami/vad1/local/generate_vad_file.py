#!/usr/bin/env python3
#
# Copyright  2023  Dongwei Jiang
# Apache 2.0

import sys
import wave

# This script use segments for ASR task to generate text for VAD task for AMI dataset.

segment_file = sys.argv[1]
text_file = sys.argv[2]
wav_file = sys.argv[3]
utt2spk_file = sys.argv[4]

segment_dict = {}

for line in open(segment_file):
    splits = line.strip().split(" ")
    segment, key, start, end = splits[0], splits[1], splits[2], splits[3]
    if key not in segment_dict:
        segment_dict[key] = []
    segment_dict[key].append((start, end))

segment_write_file = open(segment_file, "w")
text_write_file = open(text_file, "w")
utt2spk_write_file = open(utt2spk_file, "w")

for line in open(wav_file):
    splits = line.strip().split(" ")
    key, wav_name = splits[0], splits[8]
    # get the length of the wav file
    with wave.open(wav_name, "r") as wav_file:
        duration = wav_file.getnframes() / float(wav_file.getframerate())
    # split the whole wave file into 10s segments
    for i in range(int(duration / 10)):
        start = i * 10
        end = (i + 1) * 10
        # get the overlap segments
        if key in segment_dict:
            overlap_segments = []
            for segment in segment_dict[key]:
                if float(segment[0]) <= end and float(segment[1]) >= start:
                    overlap_segments.append(
                        str(round(max(float(segment[0]), start) - start, 2))
                        + " "
                        + str(round(min(float(segment[1]), end) - start, 2))
                    )
                if float(segment[0]) > end:
                    break
            if len(overlap_segments) > 0:
                # merge segments that are continuous
                merged_segments = []
                for i in range(len(overlap_segments)):
                    if i == 0:
                        merged_segments.append(overlap_segments[i])
                    else:
                        if float(overlap_segments[i].split(" ")[0]) == float(
                            merged_segments[-1].split(" ")[1]
                        ):
                            merged_segments[-1] = (
                                merged_segments[-1].split(" ")[0]
                                + " "
                                + overlap_segments[i].split(" ")[1]
                            )
                        else:
                            merged_segments.append(overlap_segments[i])
                text_write_file.write(
                    key
                    + "_"
                    + str(start).zfill(4)
                    + " "
                    + " ".join(merged_segments)
                    + "\n"
                )
                segment_write_file.write(
                    key
                    + "_"
                    + str(start).zfill(4)
                    + " "
                    + key
                    + " "
                    + str(start)
                    + " "
                    + str(end)
                    + "\n"
                )
                utt2spk_write_file.write(
                    key + "_" + str(start).zfill(4) + " " + key + "\n"
                )

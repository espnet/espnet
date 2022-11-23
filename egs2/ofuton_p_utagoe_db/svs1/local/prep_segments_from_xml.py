#!/usr/bin/env python3
import argparse
import math
import os
import sys

import music21 as m21

from espnet2.fileio.score_scp import XMLReader

"""Divide songs into segments according to structured musicXML."""

class LabelInfo(object):
    def __init__(self, start, end, label_id, midi):
        self.label_id = label_id
        self.midi = midi
        self.start = start
        self.end = end


class SegInfo(object):
    def __init__(self):
        self.segs = []
        self.start = -1
        self.end = -1

    def add(self, start, end, label, midi):
        start = float(start)
        end = float(end)
        if self.start < 0 or self.start > start:
            self.start = start
        if self.end < end:
            self.end = end
        self.segs.append((start, end, label, midi))

    def split(self, threshold=30):
        seg_num = math.ceil((self.end - self.start) / threshold)
        if seg_num == 1:
            return [self.segs]
        avg = (self.end - self.start) / seg_num
        return_seg = []

        start_time = self.start
        cache_seg = []
        for seg in self.segs:
            cache_time = seg[1] - start_time
            if cache_time > avg:
                return_seg.append(cache_seg)
                start_time = seg[0]
                cache_seg = [seg]
            else:
                cache_seg.append(seg)

        return_seg.append(cache_seg)
        return return_seg


def pack_zero(file_id, number, length=4):
    number = str(number)
    return file_id + "_" + "0" * (length - len(number)) + number


def get_parser():
    parser = argparse.ArgumentParser(
        description="Prepare segments from MUSICXML files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("scp", type=str, help="scp folder")
    parser.add_argument(
        "--threshold", type=int, help="threshold for silence identification.", default=30000
    )
    parser.add_argument(
        "--silence", action="append", help="silence_phone", default=["pau"]
    )
    return parser

def make_segment(file_id, tempo, labels, threshold, sil=["P", "B"]):
    segments = []
    segment = SegInfo()
    for label in labels:
        if label.label_id in sil:
            if len(segment.segs) > 0:
                segments.extend(segment.split(threshold=threshold))
                segment = SegInfo()
            continue
        segment.add(label.start, label.end, label.label_id, label.midi)

    if len(segment.segs) > 0:
        segments.extend(segment.split(threshold=threshold))

    segments_w_id = {}
    id = 0
    for seg in segments:
        if len(seg) == 0:
            continue
        segments_w_id[pack_zero(file_id, id)] = tempo, seg
        id += 1
    return segments_w_id


if __name__ == "__main__":
    args = get_parser().parse_args()
    args.threshold *= 1e-3
    segments = []
    musicxmlscp = open(os.path.join(args.scp, "musicxml.scp"), "r", encoding="utf-8")
    update_segments = open(
        os.path.join(args.scp, "segments_from_xml.tmp"), "w", encoding="utf-8"
    )
    update_score = open(os.path.join(args.scp, "score.tmp"), "w", encoding="utf-8")
    update_text = open(os.path.join(args.scp, "text.tmp"), "w", encoding="utf-8")
    reader = XMLReader(os.path.join(args.scp, "musicxml.scp"))
    for xml_line in musicxmlscp:
        xmlline = xml_line.strip().split(" ")
        recording_id = xmlline[0]
        path = xmlline[1]
        lyrics, notes, segs, tempo = reader[recording_id]
        temp_info = []
        for i in range(len(lyrics)):
            temp_info.append(LabelInfo(segs[i][0], segs[i][1], lyrics[i], notes[i]))
        segments.append(
            make_segment(recording_id, tempo, temp_info, args.threshold, args.silence)
        )

    for file in segments:
        for key, (tempo, val) in file.items():
            segment_begin = "{:.3f}".format(val[0][0])
            segment_end = "{:.3f}".format(val[-1][1])

            update_segments.write(
                "{} {} {} {}\n".format(
                    key, "_".join(key.split("_")[:-1]), segment_begin, segment_end
                )
            )
            update_text.write("{} ".format(key))
            update_score.write("{}  {}".format(key, tempo))
            for v in val:
                update_score.write("  {:.3f} {:.3f} {} {}".format(v[0], v[1], v[2], v[3]))
                update_text.write(" {}".format(v[2]))
            update_score.write("\n")
            update_text.write("\n")

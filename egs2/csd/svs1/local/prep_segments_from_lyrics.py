#!/usr/bin/env python3
import argparse
import math
import os
import sys

from espnet2.fileio.score_scp import MIDReader, SingingScoreWriter

class LabelInfo(object):
    def __init__(self, start, end, label_id):
        self.label_id = label_id
        self.start = start
        self.end = end


class SegInfo(object):
    def __init__(self):
        self.segs = []
        self.start = -1
        self.end = -1

    def add(self, start, end, label):
        start = float(start)
        end = float(end)
        if self.start < 0 or self.start > start:
            self.start = start
        if self.end < end:
            self.end = end
        self.segs.append((start, end, label))

    def split(self, threshold=10):
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
        description="Prepare segments from HTS-style alignment files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("scp", type=str, help="scp folder")
    parser.add_argument(
        "threshold", type=int, help="threshold for silence identification."
    )
    parser.add_argument(
        "--silence", action="append", help="silence_phone", default=["pau"]
    )
    return parser


def make_segment(file_id, labels, threshold=13.5, sil=["pau", "br", "sil"]):
    segments = []
    segment = SegInfo()
    last_label = None
    for label in labels:
        if last_label is not None and last_label.end < label.start:
            if len(segment.segs) > 0:
                segments.extend(segment.split(threshold=threshold))
                segment = SegInfo()
            continue
        segment.add(label.start, label.end, label.label_id)

    if len(segment.segs) > 0:
        segments.extend(segment.split(threshold=threshold))

    segments_w_id = {}
    id = 0
    for seg in segments:
        if len(seg) == 0:
            continue
        segments_w_id[pack_zero(file_id, id)] = seg
        id += 1
    return segments_w_id


if __name__ == "__main__":
    # os.chdir(sys.path[0]+'/..')
    args = get_parser().parse_args()
    args.threshold *= 1e-3

    seg_text = open(os.path.join(args.scp, "segments"), "r", encoding="utf-8")
    label = open(os.path.join(args.scp, "label"), "r", encoding="utf-8")
    wavscp = open(os.path.join(args.scp, "wav.scp"), "r", encoding="utf-8")

    update_segments = open(
        os.path.join(args.scp, "segments.tmp"), "w", encoding="utf-8"
    )
    update_label = open(os.path.join(args.scp, "label.tmp"), "w", encoding="utf-8")
    update_text = open(os.path.join(args.scp, "text.tmp"), "w", encoding="utf-8")

            
    # segment every line in `csd/lang/txt/*.txt`
    for wav_line in wavscp:
        wavline = wav_line.strip().split()
        recording_id = wavline[0]
        label_line = label.readline()
        phn_info = label_line.strip().split()[1:]
        len_all = len(phn_info) # 3 times
        index = 0
        while index < len_all:
            seg_line = seg_text.readline() # rid syb1 syb2 ...
            segline = seg_line.strip().split()  
            rid = segline[0]
            assert recording_id == "_".join(rid.split('_')[:-1])
            seg_txt = segline[1:] # 1 times
            temp_info = []
            update_label.write("{}".format(rid))
            update_text.write("{}".format(rid))
            for i in range(len(seg_txt)):
                v0 = phn_info[index]
                v1 = phn_info[index + 1]
                v2 = phn_info[index + 2]
                temp_info.append(
                    LabelInfo(v0, v1, v2)
                )
                update_label.write(" {} {} {}".format(v0, v1, v2))
                update_text.write(" {}".format(v2))
                index = index + 3
            segment_begin = temp_info[0].start
            segment_end = temp_info[-1].end
            update_segments.write(
                "{} {} {} {}\n".format(
                    rid, recording_id, segment_begin, segment_end
                )
            )
            update_label.write("\n")
            update_text.write("\n")
            

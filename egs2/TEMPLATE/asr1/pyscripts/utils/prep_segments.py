#!/usr/bin/env python3
import argparse
import math
import os
import sys

import music21 as m21

from espnet2.fileio.score_scp import SingingScoreWriter, XMLReader


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

    def add(self, start, end, label, midi=None):
        start = float(start)
        end = float(end)
        if self.start < 0 or self.start > start:
            self.start = start
        if self.end < end:
            self.end = end
        if midi is None:
            self.segs.append((start, end, label))
        else:
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
        description="Prepare segments from either HTS-style alignment files or MUSICXML files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("scp", type=str, help="scp folder")
    parser.add_argument(
        "--input_type",
        type=str,
        choices=["hts", "xml"],
        help="type of input files (hts for HTS-style alignment files, xml for MUSICXML files)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        help="threshold for silence identification.",
        default=30000,
    )
    parser.add_argument(
        "--silence", action="append", help="silence_phone", default=["pau"]
    )
    parser.add_argument(
        "--score_dump", type=str, default="score_dump", help="score dump directory"
    )
    args = parser.parse_args()
    return parser, args


def make_segment_hts(file_id, labels, threshold=30, sil=["pau", "br", "sil"]):
    segments = []
    segment = SegInfo()
    for label in labels:
        if label.label_id in sil:
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


def make_segment_xml(file_id, tempo, notes, threshold, sil=["P", "B"]):
    segments = []
    segment = SegInfo()
    for note in notes:
        # Divide songs by 'P' (pause) or 'B' (breath)
        if note.lyric in sil:
            if len(segment.segs) > 0:
                segments.extend(segment.split(threshold=threshold))
                segment = SegInfo()
            continue
        segment.add(note.st, note.et, note.lyric, note.midi)
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
    parser, args = get_parser()
    args.threshold *= 1e-3
    segments = []

    if args.input_type == "hts":
        file_scp = open(os.path.join(args.scp, "wav.scp"), "r", encoding="utf-8")
        label_file = open(os.path.join(args.scp, "label"), "r", encoding="utf-8")

        for file_line in file_scp:
            label_line = label_file.readline()
            if not label_line:
                raise ValueError("not match label and wav.scp in {}".format(args.scp))

            fileline = file_line.strip().split(" ")
            recording_id = fileline[0]
            path = " ".join(fileline[1:])
            phn_info = label_line.strip().split()[1:]
            temp_info = []
            for i in range(len(phn_info) // 3):
                temp_info.append(
                    LabelInfo(phn_info[i * 3], phn_info[i * 3 + 1], phn_info[i * 3 + 2])
                )
            segments.append(
                make_segment_hts(recording_id, temp_info, args.threshold, args.silence)
            )
        update_segments = open(
            os.path.join(args.scp, "segments.tmp"), "w", encoding="utf-8"
        )

    elif args.input_type == "xml":
        file_scp = open(os.path.join(args.scp, "score.scp"), "r", encoding="utf-8")
        reader = XMLReader(os.path.join(args.scp, "score.scp"))
        for xml_line in file_scp:
            xmlline = xml_line.strip().split(" ")
            recording_id = xmlline[0]
            path = xmlline[1]
            tempo, temp_info = reader[recording_id]
            segments.append(
                make_segment_xml(
                    recording_id, tempo, temp_info, args.threshold, args.silence
                )
            )
        update_segments = open(
            os.path.join(args.scp, "segments_from_xml.tmp"), "w", encoding="utf-8"
        )

    update_text = open(os.path.join(args.scp, "text.tmp"), "w", encoding="utf-8")
    update_label = open(os.path.join(args.scp, "label.tmp"), "w", encoding="utf-8")

    writer = SingingScoreWriter(
        args.score_dump, os.path.join(args.scp, "score.scp.tmp")
    )

    for file in segments:
        for key, val in file.items():
            if args.input_type == "xml":
                tempo, val = val
                score = dict(
                    tempo=tempo, item_list=["st", "et", "lyric", "midi"], note=val
                )
                writer[key] = score
            segment_begin = "{:.3f}".format(val[0][0])
            segment_end = "{:.3f}".format(val[-1][1])
            update_segments.write(
                "{} {} {} {}\n".format(
                    key, "_".join(key.split("_")[:-1]), segment_begin, segment_end
                )
            )
            update_text.write("{} ".format(key))
            update_label.write("{}".format(key))

            for v in val:
                if args.input_type == "hts":
                    update_label.write(" {:.3f} {:.3f}  {}".format(v[0], v[1], v[2]))
                update_text.write(" {}".format(v[2]))

            if args.input_type == "hts":
                update_label.write("\n")
            update_text.write("\n")

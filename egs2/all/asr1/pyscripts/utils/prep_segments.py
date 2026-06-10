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


def get_parser():
    parser = argparse.ArgumentParser(
        description="Prepare segments from either HTS-style \n"
        "alignment files or MUSICXML files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("scp", type=str, help="scp folder")
    parser.add_argument(
        "--input_type",
        type=str,
        choices=["hts", "xml"],
        help="type of input files\n"
        "(hts for HTS-style alignment files, xml for MUSICXML files)",
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


class DataHandler:
    def __init__(self, parser, args):
        self.parser, self.args = parser, args
        self.args.threshold *= 1e-3
        self.segments = []

        if self.args.input_type == "hts":
            self.file_scp = open(
                os.path.join(self.args.scp, "wav.scp"), "r", encoding="utf-8"
            )
            self.label_file = open(
                os.path.join(self.args.scp, "label"), "r", encoding="utf-8"
            )

        elif self.args.input_type == "xml":
            self.file_scp = open(
                os.path.join(self.args.scp, "score.scp"), "r", encoding="utf-8"
            )
            self.xml_reader = XMLReader(os.path.join(self.args.scp, "score.scp"))

        self.update_segments = None
        self.update_text = open(
            os.path.join(self.args.scp, "text.tmp"), "w", encoding="utf-8"
        )
        self.update_label = open(
            os.path.join(self.args.scp, "label.tmp"), "w", encoding="utf-8"
        )
        self.writer = SingingScoreWriter(
            self.args.score_dump, os.path.join(self.args.scp, "score.scp.tmp")
        )

    def replace_lyrics(self, start, lyric, labels, segment, segments):
        """
        replace wrong lyrics with correct one
        """
        labels[start].lyric = lyric
        return labels, segment, segments, False

    def replace_labels(self, start, label_id, labels, segment, segments):
        """
        replace wrong phoneme with correct one
        """
        labels[start].label_id = label_id
        return labels, segment, segments, False

    def skip_labels(self, start, labels, segment, segments):
        """
        remove wrong phoneme
        """
        labels[start + 1].start = labels[start].start
        return labels, segment, segments, True

    def add_missing_phoneme(
        self, start, label_id, time, labels, segment, segments, skip=False
    ):
        segment.add(labels[start].start, time, label_id)
        labels[start].start = time
        return labels, segment, segments, skip

    def add_pause(self, labels, segment, segments, threshold):
        segments.extend(segment.split(threshold=threshold))
        segment = SegInfo()
        return labels, segment, segments, False

    def pack_zero(self, file_id, number, length=4):
        number = str(number)
        return file_id + "_" + "0" * (length - len(number)) + number

    def get_error_dict(self, input_type=None):
        error_dict = {}
        return error_dict

    def fix_dataset(
        self, input_type, file_id, i, labels, segment, segments, threshold=30
    ):
        skip = False
        label = labels[i]
        error_dict = self.get_error_dict(input_type)

        if error_dict:
            for file_id_ in error_dict:
                if file_id_ in file_id:
                    for func in error_dict[file_id_]:
                        labels, segment, segments, skip = func(
                            i, labels, segment, segments, threshold
                        )

        return label, segment, segments, skip

    def make_segment_hts(self, file_id, labels, threshold=30, sil=["pau", "br", "sil"]):
        segments = []
        segment = SegInfo()
        for i in range(len(labels)):
            label, segment, segments, skip = self.fix_dataset(
                "hts", file_id, i, labels, segment, segments, threshold
            )
            if skip:
                continue
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
            segments_w_id[self.pack_zero(file_id, id)] = seg
            id += 1
        return segments_w_id

    def make_segment_xml(self, file_id, tempo, notes, threshold, sil=["P", "B"]):
        segments = []
        segment = SegInfo()
        for i in range(len(notes)):
            note = notes[i]
            note, segment, segments, skip = self.fix_dataset(
                "xml", file_id, i, notes, segment, segments, threshold
            )
            if skip:
                continue
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
            segments_w_id[self.pack_zero(file_id, id)] = tempo, seg
            id += 1
        return segments_w_id

    def process_files(self):
        if self.args.input_type == "hts":
            self.process_hts_files()

        elif self.args.input_type == "xml":
            self.process_xml_files()

    def process_hts_files(self):
        self.update_segments = open(
            os.path.join(self.args.scp, "segments.tmp"), "w", encoding="utf-8"
        )

        for file_line in self.file_scp:
            label_line = self.label_file.readline()
            if not label_line:
                raise ValueError(
                    "not match label and wav.scp in {}".format(self.args.scp)
                )

            fileline = file_line.strip().split(" ")
            recording_id = fileline[0]
            path = " ".join(fileline[1:])
            phn_info = label_line.strip().split()[1:]
            temp_info = []
            for i in range(len(phn_info) // 3):
                temp_info.append(
                    LabelInfo(phn_info[i * 3], phn_info[i * 3 + 1], phn_info[i * 3 + 2])
                )
            self.segments.append(
                self.make_segment_hts(
                    recording_id,
                    temp_info,
                    self.args.threshold,
                    self.args.silence,
                )
            )

    def process_xml_files(self):
        self.update_segments = open(
            os.path.join(self.args.scp, "segments_from_xml.tmp"), "w", encoding="utf-8"
        )

        for xml_line in self.file_scp:
            xmlline = xml_line.strip().split(" ")
            recording_id = xmlline[0]
            path = xmlline[1]
            tempo, temp_info = self.xml_reader[recording_id]

            self.segments.append(
                self.make_segment_xml(
                    recording_id,
                    tempo,
                    temp_info,
                    self.args.threshold,
                    self.args.silence,
                )
            )

    def write_files(self):
        for file in self.segments:
            for key, val in file.items():
                if self.args.input_type == "xml":
                    tempo, val = val
                    score = dict(
                        tempo=tempo, item_list=["st", "et", "lyric", "midi"], note=val
                    )
                    self.writer[key] = score

                segment_begin = "{:.3f}".format(val[0][0])
                segment_end = "{:.3f}".format(val[-1][1])
                self.update_segments.write(
                    "{} {} {} {}\n".format(
                        key, "_".join(key.split("_")[:-1]), segment_begin, segment_end
                    )
                )
                self.update_text.write("{} ".format(key))
                self.update_label.write("{}".format(key))

                for v in val:
                    if self.args.input_type == "hts":
                        self.update_label.write(
                            " {:.3f} {:.3f}  {}".format(v[0], v[1], v[2])
                        )
                    self.update_text.write(" {}".format(v[2]))

                if self.args.input_type == "hts":
                    self.update_label.write("\n")
                self.update_text.write("\n")


if __name__ == "__main__":
    parser, args = get_parser()
    handler = DataHandler(parser, args)
    handler.process_files()
    handler.write_files()

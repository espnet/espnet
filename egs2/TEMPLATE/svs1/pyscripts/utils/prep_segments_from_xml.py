#!/root/anaconda3/bin/python3
import argparse
import math
import os
import sys
import music21 as m21

"""Divide songs into segments according to structured musicXML."""

class LabelInfo(object):
    def __init__(self, start, end, label_id, isNote, note):
        self.label_id = label_id
        self.start = start
        self.end = end
        self.isNote = isNote
        self.note = note
    
    def extend(self, start, end, label_id):
        if self.start == self.end:
            self.label_id = label_id
            self.start = start
            self.end = end
        else:
            self.end = end


class SegInfo(object):
    def __init__(self):
        self.segs = []
        self.start = -1
        self.end = -1
        self.off = -1

    def add(self, start, end, label, note):
        start = float(start)
        end = float(end)
        if self.start < 0 or self.start > start:
            self.start = start
            self.off = note.offset
        if self.end < end:
            self.end = end
        note.offset -= self.off
        self.segs.append((start, end, label, note))

    def split(self, threshold=10):
        seg_num = math.ceil((self.end - self.start) / threshold)
        if seg_num == 1:
            return [self.segs]
        avg = (self.end - self.start) / seg_num
        return_seg = []

        start_time = self.start
        cache_seg = []
        off = self.segs[0][3].offset
        for seg in self.segs:
            cache_time = seg[1] - start_time
            if cache_time > avg:
                return_seg.append(cache_seg)
                start_time = seg[0]
                off = seg[3].offset
                seg[3].offset -= off
                cache_seg = [seg]
            else:
                seg[3].offset -= off
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
    parser.add_argument(
        "--xml_dump", type=str, default="xml_dump", help="xml dump directory")
        # TODO (Yuning): update xml prepare scheme
    
    args = parser.parse_args()
    if not os.path.exists(args.xml_dump):
        os.makedirs(args.xml_dump)

    return parser


def make_segment(file_id, labels, threshold=13.5):
    segments = []
    segment = SegInfo()
    for label in labels:
        if label.isNote == "Rest":
            if len(segment.segs) > 0:
                segments.extend(segment.split(threshold=threshold))
                segment = SegInfo()
            continue
        segment.add(label.start, label.end, label.label_id, label.note)

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
    args = get_parser().parse_args()
    args.threshold *= 1e-3
    segments = []
    musicxmlscp = open(os.path.join(args.scp, "musicxml.scp"), "r", encoding="utf-8")
    update_segments = open(
        os.path.join(args.scp, "segments_from_xml.tmp"), "w", encoding="utf-8"
    )
    update_xmlnote = open(
        os.path.join(args.scp, "xmlnote.tmp"), "w", encoding="utf-8"
    )

    for xml_line in musicxmlscp:
        xmlline = xml_line.strip().split(" ")
        recording_id = xmlline[0]
        path = xmlline[1]

        temp_info = []
        score = m21.converter.parse(path)
        part = score.parts[0].flat
        t = 0
        rest = LabelInfo(0, 0, 0, "Rest", None)
        for note in part.notesAndRests:
            if note.isNote:
                if rest.start != rest.end:
                    temp_info.append(rest)
                    rest = LabelInfo(0, 0, 0, "Rest", None)
                temp_info.append(
                    LabelInfo(t, t + note.seconds, note.offset, "Note", note)
                )
            elif note.isRest:
                rest.extend(t, t + note.seconds, note.offset)
            t += note.seconds
        if rest.start != rest.end:
            temp_info.append(rest)

        segments.append(
            make_segment(recording_id, temp_info, args.threshold)
        )

    for file in segments:
        for key, val in file.items():
            segment_begin = "{:.3f}".format(val[0][0])
            segment_end = "{:.3f}".format(val[-1][1])

            update_segments.write(
                "{} {} {} {}\n".format(
                    key, "_".join(key.split("_")[:-1]), segment_begin, segment_end
                )
            )
            update_xmlnote.write("{}".format(key))
            new_stream = m21.stream.Stream()
            for v in val:
                update_xmlnote.write(" {}".format(v[2]))
                new_stream.insert(v[3].offset, v[3])
                 
            update_xmlnote.write("\n")
            new_stream.write('xml', fp = os.path.join(args.xml_dump, key + ".musicxml"))

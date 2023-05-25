#!/usr/bin/env python3
import argparse
import math
import os
import sys

import music21 as m21

from espnet2.fileio.score_scp import SingingScoreWriter, XMLReader, MIDReader

"""Generate segments according to structured musicXML."""
"""Transfer music score (from musicXML and MIDI) into 'score.json' format."""


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
        "--threshold",
        type=int,
        help="threshold for silence identification.",
        default=30000,
    )
    parser.add_argument(
        "--silence", action="append", help="silence_phone", default=["P"]
    )
    parser.add_argument(
        "--score_dump", type=str, default="score_dump", help="score dump directory"
    )
    args = parser.parse_args()
    return parser


def make_segment(file_id, tempo, notes, threshold, sil=["P", "B"]):
    segments = []
    segment = SegInfo()
    for i in range(len(notes)):
        note = notes[i]
        # replace wrong note lyric with correct one
        if "01" in file_id and note.lyric == "え" and notes[i - 1].lyric == "の":
            note.lyric = "め"
        # add pauses
        if ("03" in file_id and ((note.lyric == "か" and notes[i - 1].lyric == "の") or (note.lyric == "と" and notes[i + 1].lyric == "な"))) or ("23" in file_id and note.lyric == "む" and notes[i - 1].lyric == "い"):
            segments.extend(segment.split(threshold=threshold))
            segment = SegInfo()

        # Divide songs by 'P' (pause) or 'B' (breath)
        if note.lyric in sil:
            # remove rest note
            if "03" in file_id and notes[i + 1].lyric == "せ":
                notes[i + 1].st = note.st
                continue
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


def align_lyric_note(recording_id, mid_info, xml_info, sil):
# NOTE(Yuning): Some XMLs cannot be used directly, we only extract lyric information
# from xmls and assign them to the notes from MIDIs
    
    # load scores from mid and xml
    mid_tempo, note_lis = mid_info
    xml_tempo, lyric_lis = xml_info
    # fix errors dataset
    # add pause into xml
    if "41" in recording_id:
        lyric_lis[49].lyric = "P"
        lyric_lis[49].st = 22.612
        lyric_lis[48].et = 22.612
    note_seq = []
    # check tempo
    if mid_tempo != xml_tempo:
        raise ValueError("Different tempo from XML and MIDI in {}.".format(recording_id))
    k = 0
    for i in range(len(note_lis)):
        note = note_lis[i]
        # skip rest notes in xml 
        while k < len(lyric_lis) and lyric_lis[k].lyric in sil:
            k += 1
        if k >= len(lyric_lis):
            raise ValueError("lyrics from XML is longer than MIDI in {}.".format(recording_id))
        # assign current lyric to note
        if note.lyric == "*":
            # NOTE(Yuning): In natsume, for lyric with 'っ', the note shouldn't be separate two 
            # notes in MIDI. Special check for midi might be added in other datasets like,
            # 'note_lis[i + 1].midi == lyric_lis[k].midi'
            if note.midi == lyric_lis[k].midi:
                if 'っ' in lyric_lis[k].lyric:# and note_lis[i + 1].midi == lyric_lis[k].midi:
                    note_dur = int((note_lis[i + 1].et - note.st) * 80 + 0.5)
                    xml_dur = int((lyric_lis[k].et - lyric_lis[k].st) * 80 + 0.5)
                    # conbine the two notes
                    if note_dur == xml_dur:
                        note_lis[i + 1].st = note.st
                        note_lis[i + 1].midi = note.midi
                        continue
                note.lyric = lyric_lis[k].lyric
                note_seq.append(note)
                k += 1
            else:
                raise ValueError("Mismatch in XML {}-th: {} and MIDI {}-th of {}.".format(k, lyric_lis[k].lyric, i, recording_id))
        else:
            # add pauses from mid.
            note_seq.append(note)
    return xml_tempo, note_seq


if __name__ == "__main__":
    args = get_parser().parse_args()
    args.threshold *= 1e-3
    segments = []
    scorescp = open(os.path.join(args.scp, "score.scp"), "r", encoding="utf-8")
    midscp = open(os.path.join(args.scp, "mid.scp"), "r", encoding="utf-8")
    update_segments = open(
        os.path.join(args.scp, "segments_from_xml.tmp"), "w", encoding="utf-8"
    )
    update_text = open(os.path.join(args.scp, "text.tmp"), "w", encoding="utf-8")
    xml_reader = XMLReader(os.path.join(args.scp, "score.scp"))
    mid_reader = MIDReader(os.path.join(args.scp, "mid.scp"), add_rest=True)
    # load score 
    for xml_line in scorescp:
        xmlline = xml_line.strip().split(" ")
        recording_id = xmlline[0]
        path = xmlline[1]
        if recording_id[-2: ] in ["32", "45", "41", "03", "25", "27", "13", "30", '35', '21', '39', '23', '51', '18', '50', '29', '16', '38', '48', '46']:
            # load score (note sequence) from mid
            mid_info = mid_reader[recording_id]
            # load score (lyric) from xml
            xml_info = xml_reader[recording_id]
            tempo, temp_info = align_lyric_note(recording_id, mid_info, xml_info, args.silence)
        else:
            tempo, temp_info = xml_reader[recording_id]
        segments.append(
            make_segment(recording_id, tempo, temp_info, args.threshold, args.silence)
        )
    
    writer = SingingScoreWriter(
        args.score_dump, os.path.join(args.scp, "score.scp.tmp")
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
            score = dict(tempo=tempo, item_list=["st", "et", "lyric", "midi"], note=val)
            writer[key] = score
            for v in val:
                update_text.write(" {}".format(v[2]))
            update_text.write("\n")

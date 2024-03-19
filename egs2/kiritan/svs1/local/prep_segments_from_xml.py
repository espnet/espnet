#!/usr/bin/env python3
import argparse
import math
import os
import sys

import music21 as m21
import numpy as np
from typeguard import typechecked

from espnet2.fileio.read_text import read_2columns_text
from espnet2.fileio.score_scp import NOTE, SingingScoreWriter

"""Generate segments according to structured musicXML."""
"""Transfer music score (from musicXML and MIDI) into 'score.json' format."""


def split_lyric(lr):
    # split lyrics into single syllable
    lrs = []
    i = 0
    while i < len(lr):
        lrs.append(lr[i])
        i += 1
        while i < len(lr) and lr[i] in [
            "っ",
            "ゃ",
            "ょ",
            "ゅ",
            "ぁ",
            "ぃ",
            "ぅ",
            "ぇ",
            "ぉ",
            "゛",
        ]:
            lrs[-1] += lr[i]
            i += 1
    return lrs


class XMLReader:
    """Reader class for 'xml.scp'.

    Examples:
        key1 /some/path/a.xml
        key2 /some/path/b.xml
        key3 /some/path/c.xml
        key4 /some/path/d.xml
        ...

        >>> reader = XMLScpReader('xml.scp')
        >>> tempo, note_list = reader['key1']
    """

    @typechecked
    def __init__(
        self,
        fname,
        dtype=np.int16,
    ):
        self.fname = fname
        self.dtype = dtype
        self.data = read_2columns_text(fname)  # get key-value dict

    def __getitem__(self, key):
        score = m21.converter.parse(self.data[key])
        m = score.metronomeMarkBoundaries()
        tempo = int(m[0][2].number)
        part = score.parts[0].flat
        notes_list = []
        prepitch = -1
        st = 0
        for note in part.notesAndRests:
            dur = note.seconds
            if not note.isRest:  # Note or Chord
                lr = note.lyric
                if lr is None or lr == "" or lr == "ー":  # multi note in one syllable
                    if note.pitch.midi == prepitch:  # same pitch
                        notes_list[-1].et += dur
                    else:  # different pitch
                        notes_list.append(NOTE("—", note.pitch.midi, st, st + dur))
                else:  # normal note for one syllable
                    lrs = split_lyric(lr)  # split lyrics into single syllable
                    for syb in lrs:
                        notes_list.append(NOTE(syb, note.pitch.midi, st, st + dur))
                prepitch = note.pitch.midi
                for arti in note.articulations:  # <br> is tagged as a notation
                    if arti.name in ["breath mark"]:  # up-bow?
                        notes_list.append(NOTE("B", 0, st, st))  # , 0))
            else:  # rest note
                if prepitch == 0:
                    notes_list[-1].et += dur
                else:
                    notes_list.append(NOTE("P", 0, st, st + dur))
                prepitch = 0
            st += dur
        # NOTE(Yuning): implicit rest at the end of xml file should be removed.
        if notes_list[-1].midi == 0 and notes_list[-1].lyric == "P":
            notes_list.pop()
        return tempo, notes_list

    def get_path(self, key):
        return self.data[key]

    def __contains__(self, item):
        return item

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def keys(self):
        return self.data.keys()


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
    # add missing lyric
    if "29" in file_id:
        dur = (notes[-1].et - notes[-1].st) / 2
        notes.append(NOTE("ん", notes[-1].midi, notes[-1].st + dur, notes[-1].et))
        notes[-1].et -= dur
    for i in range(len(notes)):
        note = notes[i]
        # add pause (split)
        if len(segment.segs) and (
            ("18" in file_id and note.lyric == "ど" and notes[i - 1].lyric == "す")
            or ("28" in file_id and i > 0 and notes[i - 1].lyric == "しょ")
            or ("42" in file_id and note.lyric == "し" and notes[i - 1].lyric == "い")
            or ("21" in file_id and (note.lyric == "ぜっ" or note.lyric == "きっ"))
        ):
            segments.extend(segment.split(threshold=threshold))
            segment = SegInfo()

        # Divide songs by 'P' (pause) or 'B' (breath)
        if note.lyric in sil:
            # remove rest note
            if ("42" in file_id and i > 0 and notes[i - 1].lyric == "ぐっ") or (
                "21" in file_id
                and notes[i + 1].lyric == "し"
                and notes[i + 1].lyric == "た"
            ):
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


if __name__ == "__main__":
    args = get_parser().parse_args()
    args.threshold *= 1e-3
    segments = []
    scorescp = open(os.path.join(args.scp, "score.scp"), "r", encoding="utf-8")
    update_segments = open(
        os.path.join(args.scp, "segments_from_xml.tmp"), "w", encoding="utf-8"
    )
    update_text = open(os.path.join(args.scp, "text.tmp"), "w", encoding="utf-8")
    xml_reader = XMLReader(os.path.join(args.scp, "score.scp"))
    # load score
    for xml_line in scorescp:
        xmlline = xml_line.strip().split(" ")
        recording_id = xmlline[0]
        path = xmlline[1]
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

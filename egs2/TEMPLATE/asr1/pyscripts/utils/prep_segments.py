#!/usr/bin/env python3
import argparse
import math
import os
import sys

import music21 as m21

from espnet2.fileio.score_scp import MIDReader, SingingScoreWriter, XMLReader


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
        description="Prepare segments from either HTS-style \n"
        "alignment files or MUSICXML files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("scp", type=str, help="scp folder")
    parser.add_argument("--dataset", type=str, help="dataset name")
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


def make_segment_hts(dataset, file_id, labels, threshold=30, sil=["pau", "br", "sil"]):
    segments = []
    segment = SegInfo()
    for i in range(len(labels)):
        label, segment, segments, skip = fix_dataset(
            dataset, "hts", file_id, i, labels, segment, segments, threshold
        )
        if skip:
            continue
        if label.label_id in sil:
            label, skip = fix_dataset2(dataset, "hts", file_id, i, labels)
            if skip:
                continue
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


def make_segment_xml(dataset, file_id, tempo, notes, threshold, sil=["P", "B"]):
    segments = []
    segment = SegInfo()
    for i in range(len(notes)):
        note = notes[i]
        note, segment, segments, skip = fix_dataset(
            dataset, "xml", file_id, i, notes, segment, segments, threshold
        )
        if skip:
            continue
        # Divide songs by 'P' (pause) or 'B' (breath)
        if note.lyric in sil:
            label, skip = fix_dataset2(dataset, "xml", file_id, i, notes)
            if skip:
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


def replace_lyrics(start, lyric, labels, segment, segments):
    """
    replace wrong lyrics with correct one
    """
    labels[start].lyric = lyric
    return labels, segment, segments, False


def replace_labels(start, label_id, labels, segment, segments):
    """
    replace wrong phoneme with correct one
    """
    labels[start].label_id = label_id
    return labels, segment, segments, False


def skip_labels(start, labels, segment, segments):
    """
    remove wrong phoneme
    """
    labels[start + 1].start = labels[start].start
    return labels, segment, segments, True


def add_missing_phoneme(start, label_id, time, labels, segment, segments, skip=False):
    segment.add(labels[start].start, time, label_id)
    labels[start].start = time
    return labels, segment, segments, skip


def add_pause(labels, segment, segments, threshold):
    segments.extend(segment.split(threshold=threshold))
    segment = SegInfo()
    return labels, segment, segments, False


def fix_dataset(
    dataset, input_type, file_id, i, labels, segment, segments, threshold=30
):
    skip = False
    label = labels[i]
    error_correction = {}
    lyric_correction = {}
    if dataset == "natsume" and input_type == "hts":
        # dictionary mapping file_id to function calls
        error_correction = {
            "01": [
                lambda i, labels, segment, segments, threshold: replace_labels(
                    i, "a", labels, segment, segments
                )
                if labels[i].label_id == "cl" and labels[i + 1].label_id == "s"
                else (labels, segment, segments, False),
                lambda i, labels, segment, segments, threshold: skip_labels(
                    i, labels, segment, segments
                )
                if labels[i].label_id == "e" and labels[i + 1].label_id == "e"
                else (labels, segment, segments, False),
            ],
            "03": [
                lambda i, labels, segment, segments, threshold: replace_labels(
                    i, "z", labels, segment, segments
                )
                if labels[i].label_id == "s"
                and labels[i - 1].label_id == "o"
                and labels[i - 2].label_id == "o"
                else (labels, segment, segments, False),
                lambda i, labels, segment, segments, threshold: add_pause(
                    labels, segment, segments, threshold
                )
                if (labels[i].label_id == "m" and labels[i + 1].label_id == "e")
                or (labels[i].label_id == "t" and labels[i + 2].label_id == "d")
                else (labels, segment, segments, False),
            ],
            "50": [
                lambda i, labels, segment, segments, threshold: skip_labels(
                    i, labels, segment, segments
                )
                if labels[i].label_id == "o" and labels[i + 1].label_id == "a"
                else (labels, segment, segments, False),
            ],
            "08": [
                lambda i, labels, segment, segments, threshold: skip_labels(
                    i, labels, segment, segments
                )
                if labels[i].label_id == "w" and labels[i - 1].label_id == "e"
                else (labels, segment, segments, False),
            ],
            "41": [
                lambda i, labels, segment, segments, threshold: replace_labels(
                    i + 1, "a", labels, segment, segments
                )
                if labels[i].label_id == "a" and labels[i + 1].label_id == "o"
                else (labels, segment, segments, False),
                # Note that we already changed labels[i + 1] to "a".
                # So the if condition is different from the previous one.
                lambda i, labels, segment, segments, threshold: skip_labels(
                    i, labels, segment, segments
                )
                if labels[i].label_id == "a" and labels[i + 1].label_id == "a"
                else (labels, segment, segments, False),
            ],
            "10": [
                lambda i, labels, segment, segments, threshold: add_missing_phoneme(
                    i, "a", 81.00, labels, segment, segments
                )
                if labels[i].label_id == "a"
                and labels[i + 1].label_id == "a"
                and labels[i + 2].label_id == "o"
                else (labels, segment, segments, False),
            ],
        }

    if error_correction:
        for file_id_ in error_correction:
            if file_id_ in file_id:
                for func in error_correction[file_id_]:
                    labels, segment, segments, skip = func(
                        i, labels, segment, segments, threshold
                    )

    if dataset == "natsume" and input_type == "xml":
        # dictionary mapping file_id to function calls
        lyric_correction = {
            "01": [
                lambda i, labels, segment, segments, threshold: replace_lyrics(
                    i, "め", labels, segment, segments
                )
                if labels[i].lyric == "え" and labels[i - 1].lyric == "の"
                else (labels, segment, segments, False),
            ],
            "03": [
                lambda i, labels, segment, segments, threshold: add_pause(
                    labels, segment, segments, threshold
                )
                if (labels[i].lyric == "か" and labels[i - 1].lyric == "の")
                or (labels[i].lyric == "と" and labels[i + 1].lyric == "な")
                else (labels, segment, segments, False),
            ],
            "23": [
                lambda i, labels, segment, segments, threshold: add_pause(
                    labels, segment, segments, threshold
                )
                if labels[i].lyric == "む" and labels[i - 1].lyric == "い"
                else (labels, segment, segments, False),
            ],
        }

    if lyric_correction:
        for file_id_ in lyric_correction:
            if file_id_ in file_id:
                for func in lyric_correction[file_id_]:
                    labels, segment, segments, skip = func(
                        i, labels, segment, segments, threshold
                    )

    return label, segment, segments, skip


def fix_dataset2(dataset, input_type, file_id, i, labels):
    skip = False
    label = labels[i]
    error_correction = {}
    if dataset == "natsume" and input_type == "hts":
        error_correction = {
            "12": [
                lambda: skip_labels(i, labels, None, None)
                if labels[i + 1].label_id == "m" and labels[i + 2].label_id == "o"
                else (labels, None, None, False),
            ],
            "31": [
                lambda: skip_labels(i, labels, None, None)
                if labels[i + 1].label_id == "s"
                else (labels, None, None, False),
            ],
            "26": [
                lambda: skip_labels(i, labels, None, None)
                if labels[i + 1].label_id == "o"
                else (labels, None, None, False),
            ],
            "10": [
                lambda: skip_labels(i, labels, None, None)
                if labels[i + 1].label_id == "k" and labels[i + 2].label_id == "i"
                else (labels, None, None, False),
            ],
            "24": [
                lambda: skip_labels(i, labels, None, None)
                if i == 389
                else (labels, None, None, False),
            ],
            "07": [
                lambda: skip_labels(i, labels, None, None)
                if labels[i + 1].label_id == "m" and labels[i - 1].label_id == "o"
                else (labels, None, None, False),
            ],
        }

    if error_correction and i < len(labels) - 1:
        for file_id_ in error_correction:
            if file_id_ in file_id:
                for func in error_correction[file_id_]:
                    labels, _, _, skip = func()

    if dataset == "natsume" and input_type == "xml":
        # remove rest note
        if "03" in file_id and labels[i + 1].lyric == "せ":
            labels[i + 1].st = label.st
            skip = True

    return label, skip


def fix_dataset3(dataset, input_type, phn_info, i):
    if dataset == "natsume" and input_type == "hts":
        if phn_info[i * 3 + 2] == "U":
            phn_info[i * 3 + 2] = "u"
        if phn_info[i * 3 + 2] == "I":
            phn_info[i * 3 + 2] = "i"
    return phn_info


def fix_dataset4(dataset, recording_id, mid_reader, xml_reader):
    if dataset == "natsume":
        if recording_id[-2:] in [
            "32",
            "45",
            "41",
            "03",
            "25",
            "27",
            "13",
            "30",
            "35",
            "21",
            "39",
            "23",
            "51",
            "18",
            "50",
            "29",
            "16",
            "38",
            "48",
            "46",
        ]:
            # load score (note sequence) from mid
            mid_info = mid_reader[recording_id]
            # load score (lyric) from xml
            xml_info = xml_reader[recording_id]
            tempo, temp_info = align_lyric_note(
                recording_id, mid_info, xml_info, args.silence
            )
        else:
            tempo, temp_info = xml_reader[recording_id]
    else:
        raise ValueError("Dataset {} not supported.".format(dataset))
    return tempo, temp_info


def align_lyric_note(recording_id, mid_info, xml_info, sil):
    # NOTE(Yuning): Some XMLs cannot be used directly, we only extract
    # lyric information from xmls and assign them to the notes from MIDIs

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
        raise ValueError(
            "Different tempo from XML and MIDI in {}.".format(recording_id)
        )
    k = 0
    for i in range(len(note_lis)):
        note = note_lis[i]
        # skip rest notes in xml
        while k < len(lyric_lis) and lyric_lis[k].lyric in sil:
            k += 1
        if k >= len(lyric_lis):
            raise ValueError(
                "lyrics from XML is longer than MIDI in {}.".format(recording_id)
            )
        # assign current lyric to note
        if note.lyric == "*":
            # NOTE(Yuning): In natsume, for lyric with 'っ', the note shouldn't be
            # separate two notes in MIDI. Special check for midi might be added in
            # other datasets like, 'note_lis[i + 1].midi == lyric_lis[k].midi'
            if note.midi == lyric_lis[k].midi:
                if (
                    "っ" in lyric_lis[k].lyric
                ):  # and note_lis[i + 1].midi == lyric_lis[k].midi:
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
                raise ValueError(
                    "Mismatch in XML {}-th: {} and MIDI {}-th of {}.".format(
                        k, lyric_lis[k].lyric, i, recording_id
                    )
                )
        else:
            # add pauses from mid.
            note_seq.append(note)
    return xml_tempo, note_seq


if __name__ == "__main__":
    parser, args = get_parser()
    args.threshold *= 1e-3
    segments = []
    dataset = args.dataset

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
                phn_info = fix_dataset3(dataset, "hts", phn_info, i)
                temp_info.append(
                    LabelInfo(phn_info[i * 3], phn_info[i * 3 + 1], phn_info[i * 3 + 2])
                )
            segments.append(
                make_segment_hts(
                    dataset, recording_id, temp_info, args.threshold, args.silence
                )
            )
        update_segments = open(
            os.path.join(args.scp, "segments.tmp"), "w", encoding="utf-8"
        )

    elif args.input_type == "xml":
        file_scp = open(os.path.join(args.scp, "score.scp"), "r", encoding="utf-8")
        xml_reader = XMLReader(os.path.join(args.scp, "score.scp"))
        if dataset == "natsume":
            mid_reader = MIDReader(os.path.join(args.scp, "mid.scp"), add_rest=True)
        for xml_line in file_scp:
            xmlline = xml_line.strip().split(" ")
            recording_id = xmlline[0]
            path = xmlline[1]
            if dataset == "natsume":
                tempo, temp_info = fix_dataset4(
                    dataset, recording_id, mid_reader, xml_reader
                )
            else:
                tempo, temp_info = xml_reader[recording_id]

            segments.append(
                make_segment_xml(
                    dataset,
                    recording_id,
                    tempo,
                    temp_info,
                    args.threshold,
                    args.silence,
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

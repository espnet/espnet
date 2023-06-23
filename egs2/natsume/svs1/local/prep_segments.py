#!/usr/bin/env python3
import os

from egs2.TEMPLATE.svs1.pyscripts.utils.prep_segments import (
    DataHandler,
    LabelInfo,
    SegInfo,
    get_parser,
)
from espnet2.fileio.score_scp import MIDReader


class NatsumeDataHandler(DataHandler):
    def get_error_dict(self, input_type):
        error_dict = {}
        if input_type == "hts":
            error_dict = {
                "01": [
                    lambda i, labels, segment, segments, threshold: self.replace_labels(
                        i, "a", labels, segment, segments
                    )
                    if labels[i].label_id == "cl" and labels[i + 1].label_id == "s"
                    else (labels, segment, segments, False),
                    lambda i, labels, segment, segments, threshold: self.skip_labels(
                        i, labels, segment, segments
                    )
                    if labels[i].label_id == "e" and labels[i + 1].label_id == "e"
                    else (labels, segment, segments, False),
                ],
                "03": [
                    lambda i, labels, segment, segments, threshold: self.replace_labels(
                        i, "z", labels, segment, segments
                    )
                    if labels[i].label_id == "s"
                    and labels[i - 1].label_id == "o"
                    and labels[i - 2].label_id == "o"
                    else (labels, segment, segments, False),
                    lambda i, labels, segment, segments, threshold: self.add_pause(
                        labels, segment, segments, threshold
                    )
                    if (labels[i].label_id == "m" and labels[i + 1].label_id == "e")
                    or (labels[i].label_id == "t" and labels[i + 2].label_id == "d")
                    else (labels, segment, segments, False),
                ],
                "50": [
                    lambda i, labels, segment, segments, threshold: self.skip_labels(
                        i, labels, segment, segments
                    )
                    if labels[i].label_id == "o" and labels[i + 1].label_id == "a"
                    else (labels, segment, segments, False),
                ],
                "08": [
                    lambda i, labels, segment, segments, threshold: self.skip_labels(
                        i, labels, segment, segments
                    )
                    if labels[i].label_id == "w" and labels[i - 1].label_id == "e"
                    else (labels, segment, segments, False),
                ],
                "41": [
                    lambda i, labels, segment, segments, threshold: self.replace_labels(
                        i + 1, "a", labels, segment, segments
                    )
                    if labels[i].label_id == "a" and labels[i + 1].label_id == "o"
                    else (labels, segment, segments, False),
                    # Note that we already changed labels[i + 1] to "a".
                    # So the if condition is different from the previous one.
                    lambda i, labels, segment, segments, threshold: self.skip_labels(
                        i, labels, segment, segments
                    )
                    if labels[i].label_id == "a" and labels[i + 1].label_id == "a"
                    else (labels, segment, segments, False),
                ],
                "10": [
                    lambda i, labels, segment, segments, threshold: (
                        self.add_missing_phoneme(
                            i, "a", 81.00, labels, segment, segments
                        )
                        if labels[i].label_id == "a"
                        and labels[i + 1].label_id == "a"
                        and labels[i + 2].label_id == "o"
                        else (labels, segment, segments, False)
                    ),
                ],
            }
        elif input_type == "xml":
            error_dict = {
                "01": [
                    lambda i, labels, segment, segments, threshold: self.replace_lyrics(
                        i, "め", labels, segment, segments
                    )
                    if labels[i].lyric == "え" and labels[i - 1].lyric == "の"
                    else (labels, segment, segments, False),
                ],
                "03": [
                    lambda i, labels, segment, segments, threshold: self.add_pause(
                        labels, segment, segments, threshold
                    )
                    if (labels[i].lyric == "か" and labels[i - 1].lyric == "の")
                    or (labels[i].lyric == "と" and labels[i + 1].lyric == "な")
                    else (labels, segment, segments, False),
                ],
                "23": [
                    lambda i, labels, segment, segments, threshold: self.add_pause(
                        labels, segment, segments, threshold
                    )
                    if labels[i].lyric == "む" and labels[i - 1].lyric == "い"
                    else (labels, segment, segments, False),
                ],
            }
        return error_dict

    def fix_dataset2(self, input_type, file_id, i, labels):
        skip = False
        label = labels[i]
        error_correction = {}
        if input_type == "hts":
            error_correction = {
                "12": [
                    lambda: self.skip_labels(i, labels, None, None)
                    if labels[i + 1].label_id == "m" and labels[i + 2].label_id == "o"
                    else (labels, None, None, False),
                ],
                "31": [
                    lambda: self.skip_labels(i, labels, None, None)
                    if labels[i + 1].label_id == "s"
                    else (labels, None, None, False),
                ],
                "26": [
                    lambda: self.skip_labels(i, labels, None, None)
                    if labels[i + 1].label_id == "o"
                    else (labels, None, None, False),
                ],
                "10": [
                    lambda: self.skip_labels(i, labels, None, None)
                    if labels[i + 1].label_id == "k" and labels[i + 2].label_id == "i"
                    else (labels, None, None, False),
                ],
                "24": [
                    lambda: self.skip_labels(i, labels, None, None)
                    if i == 389
                    else (labels, None, None, False),
                ],
                "07": [
                    lambda: self.skip_labels(i, labels, None, None)
                    if labels[i + 1].label_id == "m" and labels[i - 1].label_id == "o"
                    else (labels, None, None, False),
                ],
            }

        if error_correction and i < len(labels) - 1:
            for file_id_ in error_correction:
                if file_id_ in file_id:
                    for func in error_correction[file_id_]:
                        labels, _, _, skip = func()

        if input_type == "xml":
            # remove rest note
            if "03" in file_id and labels[i + 1].lyric == "せ":
                labels[i + 1].st = label.st
                skip = True

        return label, skip

    def fix_dataset3(self, input_type, phn_info, i):
        if input_type == "hts":
            if phn_info[i * 3 + 2] == "U":
                phn_info[i * 3 + 2] = "u"
            if phn_info[i * 3 + 2] == "I":
                phn_info[i * 3 + 2] = "i"
        return phn_info

    def fix_dataset4(self, recording_id, mid_reader, xml_reader):
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
            tempo, temp_info = self.align_lyric_note(
                recording_id, mid_info, xml_info, args.silence
            )
        else:
            tempo, temp_info = xml_reader[recording_id]
        return tempo, temp_info

    def align_lyric_note(self, recording_id, mid_info, xml_info, sil):
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
                # for natsume
                label, skip = self.fix_dataset2("hts", file_id, i, labels)
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
                # for natsume
                label, skip = self.fix_dataset2("xml", file_id, i, notes)
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
            segments_w_id[self.pack_zero(file_id, id)] = tempo, seg
            id += 1
        return segments_w_id

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
                # for natsume
                phn_info = self.fix_dataset3("hts", phn_info, i)
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

        # for natsume
        mid_reader = MIDReader(os.path.join(self.args.scp, "mid.scp"), add_rest=True)
        for xml_line in self.file_scp:
            xmlline = xml_line.strip().split(" ")
            recording_id = xmlline[0]
            path = xmlline[1]

            # for natsume
            tempo, temp_info = self.fix_dataset4(
                recording_id, mid_reader, self.xml_reader
            )

            self.segments.append(
                self.make_segment_xml(
                    recording_id,
                    tempo,
                    temp_info,
                    self.args.threshold,
                    self.args.silence,
                )
            )


if __name__ == "__main__":
    parser, args = get_parser()
    handler = NatsumeDataHandler(parser, args)
    handler.process_files()
    handler.write_files()

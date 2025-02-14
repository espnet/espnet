#!/usr/bin/env python3
import argparse
import math
import os
import sys

from espnet2.fileio.score_scp import SingingScoreReader, SingingScoreWriter
from espnet2.text.build_tokenizer import build_tokenizer

"""Generate segments according to label."""


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
        description="Prepare segments from HTS-style alignment files",
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
        "--silence", action="append", help="silence_phone", default=["pau"]
    )
    parser.add_argument("--g2p", type=str, help="g2p", default="pyopenjtalk")
    parser.add_argument(
        "--score_dump", type=str, default="score_dump", help="score dump directory"
    )
    parser.add_argument(
        "--customed_dic",
        type=str,
        help="customed g2p for alignment at phoneme level",
        default="local/customed_dic.scp",
    )
    return parser


def make_segment(xml_writer, file_id, labels, threshold=30, sil=["pau", "br", "sil"]):
    xml_reader = SingingScoreReader(os.path.join(args.scp, "score.scp"))
    segments = []
    segment = SegInfo()
    for i in range(len(labels)):
        # remove unlabeled part
        if "08" in file_id and i < 135:
            continue
        label = labels[i]
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
        while len(seg) > 0:
            key = pack_zero(file_id, id)
            score = xml_reader[key]
            val, seg, notes = compare(key, score["note"], seg)
            segments_w_id[key] = val
            score["note"] = notes
            score["item_list"].append("phn")
            xml_writer[key] = score
            id += 1
    return segments_w_id


def load_customed_dic(file):
    """If syllable-to-phone tranlation differs from g2p,"""
    """ customed tranlation can be added to customed_dic."""
    customed_dic = {}
    with open(file, "r", encoding="utf-8") as f:
        content = f.read().strip().split("\n")
        for key in content:
            key = key.split(" ")
            customed_dic[key[0]] = key[1].split("_")
    return customed_dic


def compare(key, score, label):
    # follow pause in xml
    customed_dic = load_customed_dic(args.customed_dic)
    tokenizer = build_tokenizer(
        token_type="phn",
        bpemodel=None,
        delimiter=None,
        space_symbol="<space>",
        non_linguistic_symbols=None,
        g2p_type=args.g2p,
    )
    index = 0
    val = []
    for i in range(len(score)):
        syb = score[i][2]
        if syb == "—":
            if index > len(label):
                raise ValueError("Lyrics are longer than phones in {}".format(key))
            if (
                index == len(label)
                or label[index][2] != pre_phn
                or (
                    label[index][2] == pre_phn
                    and pre_phn == tokenizer.g2p(score[i + 1][2])[0]
                )
            ):
                dur = (val[-1][1] - val[-1][0]) / 2
                val.append((val[-1][0] + dur, val[-1][1]))
                val[-1] = (val[-1][0], val[-1][0] + dur, pre_phn)
                score[i].append(pre_phn)
                continue
        # Translate syllable into phones through g2p
        phns = tokenizer.g2p(syb)
        # In some case, translation can be different
        if syb in customed_dic:
            phns = customed_dic[syb]
        score[i].append("_".join(phns))
        pre_phn = phns[-1]
        for p in phns:
            if index >= len(label):
                raise ValueError("Lyrics are longer than phones in {}".format(key))
            elif label[index][2] == p:
                val.append(label[index])
                index += 1
            else:
                raise ValueError(
                    "Mismatch in syllable [{}]->{} and {}-th phones '{}' in {}.".format(
                        syb, phns, index, label[index][2], key
                    )
                )
    val_rest = []
    if index != len(label):
        val_rest = label[index:]
    return val, val_rest, score


if __name__ == "__main__":
    args = get_parser().parse_args()
    args.threshold *= 1e-3
    segments = []

    wavscp = open(os.path.join(args.scp, "wav.scp"), "r", encoding="utf-8")
    label = open(os.path.join(args.scp, "label"), "r", encoding="utf-8")

    update_segments = open(
        os.path.join(args.scp, "segments.tmp"), "w", encoding="utf-8"
    )
    update_label = open(os.path.join(args.scp, "label.tmp"), "w", encoding="utf-8")
    xml_writer = SingingScoreWriter(
        args.score_dump, os.path.join(args.scp, "score.scp.tmp")
    )

    for wav_line in wavscp:
        label_line = label.readline()
        if not label_line:
            raise ValueError("not match label and wav.scp in {}".format(args.scp))

        wavline = wav_line.strip().split(" ")
        recording_id = wavline[0]
        path = " ".join(wavline[1:])
        phn_info = label_line.strip().split()[1:]
        temp_info = []
        for i in range(len(phn_info) // 3):
            temp_info.append(
                LabelInfo(phn_info[i * 3], phn_info[i * 3 + 1], phn_info[i * 3 + 2])
            )
        segments.append(
            make_segment(
                xml_writer, recording_id, temp_info, args.threshold, args.silence
            )
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
            update_label.write("{}".format(key))

            for v in val:
                update_label.write(" {:.3f} {:.3f}  {}".format(v[0], v[1], v[2]))
            update_label.write("\n")

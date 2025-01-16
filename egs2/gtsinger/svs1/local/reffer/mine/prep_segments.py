#!/usr/bin/env python3
import argparse
import math
import os
import sys

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
    return parser


#file_id：当前音频文件的标识符，用于关联特定文件
#labels：标签的列表，每个标签通常包含: 起始时间、结束时间、标签的标识符（如音素或特定字符）
#threshold（默认值为 30）：一个时间阈值，用于限制片段的最大长度
#sil（默认值为 ["pau", "br", "sil"]）：表示沉默或暂停的音素标签，当遇到这些标签时，当前片段会被结束

def make_segment(file_id, labels, threshold=30, sil=["pau", "br", "sil"]):
    segments = []
    segment = SegInfo()
    for i in range(len(labels)):
        label = labels[i]
        if label.label_id in sil:
            if len(segment.segs) > 0:
                segments.extend(segment.split(threshold=threshold))
                segment = SegInfo()
            continue
        # add pause (split)
        if (
            (
                "turkey_in_the_straw" in file_id
                and label.label_id == "s"
                and labels[i + 1].label_id == "e"
                and labels[i + 2].label_id == "N"
            )
            or (
                "yuki" in file_id
                and label.label_id == "w"
                and labels[i + 1].label_id == "a"
                and labels[i + 2].label_id == "t"
            )
            or (
                "alps_ichimanjaku" in file_id
                and label.label_id == "a"
                and labels[i - 1].label_id == "e"
            )
        ):
            segments.extend(segment.split(threshold=threshold))
            segment = SegInfo()
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
    args = get_parser().parse_args()
    args.threshold *= 1e-3
    segments = []
    
    #args.scp = data/tr_no_dev
    wavscp = open(os.path.join(args.scp, "wav.scp"), "r", encoding="utf-8")
    label = open(os.path.join(args.scp, "label"), "r", encoding="utf-8")

    #处理文件与歌手之间的关系
    update_utt2spk = open(os.path.join(args.scp, "utt2spk.temp"), "w", encoding="utf-8")
    wav_singer_dict = {}
    with open(os.path.join(args.scp, "utt2spk"), "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()  # 去除行首和行尾的空白字符
            if line:  # 如果行不为空
                parts = line.split()  # 按空格分割
                if len(parts) == 2:  # 确保行中有两部分
                    key, value = parts
                    wav_singer_dict[key] = value
   
    update_segments = open(
        os.path.join(args.scp, "segments.tmp"), "w", encoding="utf-8"
    )
    update_label = open(os.path.join(args.scp, "label.tmp"), "w", encoding="utf-8")
   
    #wav_line = GTSINGER_Tenor1_Glissando_我的歌声里_0000 wav_dump/Tenor1_Glissando_我的歌声里_0000_bits16.wav
    for wav_line in wavscp:
        #label_line = GTSINGER_Tenor1_Glissando_我的歌声里_0000 0.0 0.22 n 0.22 0.6 i 0.6 0.7 d 0.7 1.58 e ...
        label_line = label.readline()
        if not label_line:
            raise ValueError("not match label and wav.scp in {}".format(args.scp))

        wavline = wav_line.strip().split(" ")
        recording_id = wavline[0] #GTSINGER_Tenor1_Glissando_我的歌声里_0000
        path = " ".join(wavline[1:]) #wav_dump/Tenor1_Glissando_我的歌声里_0000_bits16.wav
        phn_info = label_line.strip().split()[1:] #0.0 0.22 n 0.22 0.6 i 0.6 0.7 d 0.7 1.58 e ...
        temp_info = []
        # correct capitalization
        for i in range(len(phn_info) // 3):
            if phn_info[i * 3 + 2] == "U":
                phn_info[i * 3 + 2] = "u"
            temp_info.append(
                LabelInfo(phn_info[i * 3], phn_info[i * 3 + 1], phn_info[i * 3 + 2])
            )
        #生成音频片段
        new_segments = make_segment(recording_id, temp_info, args.threshold, args.silence)
        segments.append(new_segments)

        for key in new_segments.keys():
            update_utt2spk.write("{} {}\n".format(key,wav_singer_dict[wav_line.split(" ")[0]]))
    for file in segments:
        for key, val in file.items():
            segment_begin = "{:.3f}".format(val[0][0])
            segment_end = "{:.3f}".format(val[-1][1])

            #GTSINGER_Tenor1_Glissando_我的歌声里_0001_0000 (key)
            #GTSINGER_Tenor1_Glissando_我的歌声里_0001
            #0.000 5.740
            update_segments.write(
                "{} {} {} {}\n".format(
                    key, "_".join(key.split("_")[:-1]), segment_begin, segment_end
                )
            )
            update_label.write("{}".format(key))

            for v in val:
                update_label.write(" {:.3f} {:.3f}  {}".format(v[0], v[1], v[2]))
            update_label.write("\n")

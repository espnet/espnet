#!/usr/bin/env python3
import argparse
import math
import os
import sys

import music21 as m21

from espnet2.fileio.score_scp import SingingScoreWriter, XMLReader

"""Generate segments according to structured musicXML."""
"""Transfer music score (from musicXML) into 'score.json' format."""


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


#file_id: 当前音频文件的标识符: GTSINGER_Tenor1_Glissando_我的歌声里_0000
#tempo: 音频的节奏
#notes: 这个列表包含了音符信息，每个音符有多个属性，如开始时间（st）、结束时间（et）、歌词（lyric）等, [<espnet2.fileio.score_scp.NOTE object at 0x7f25077913d0>,...]
#threshold: 这是段落的最大时间阈值，用于决定何时分割段落。单位一般是秒或毫秒
#sil: 这是一个暂停标签的列表，默认为 ["P", "B"]，表示常见的暂停和呼吸
def make_segment(file_id, tempo, notes, threshold, sil=["P", "B"]):
    segments = []
    segment = SegInfo()
    i, length = 0, len(notes)
    while i < length:
        # Divide songs by 'P' (pause) or 'B' (breath) or GlottalStop
        #note[i] = <espnet2.fileio.score_scp.NOTE object at 0x7f07c26048b0>
        #note[i]包括lyric、midi、st、et: 你|58|3.780|4.146    P|0|5.365|5.731
        # fix errors in dataset
        # remove rest note
        # P:
        # -:表示歌词延续
        # SP是silence，AP是换气声（就吸一口气）        
        if notes[i].lyric == 'P':
            if notes[i].et - notes[i].st < 1:
                notes[i].lyric = 'AP'
            else:
                notes[i].lyric = 'SP'
            #i += 1
            #continue
        '''
        if i < (length - 1) and notes[i+1].lyric == '—':#歌词连续
            ref = i+1
            while ref<length and notes[ref].lyric == '—':
                notes[i].et = notes[ref].et  
                notes.pop(ref) 
                length -= 1
        '''
        if i < length:
            segment.add(notes[i].st, notes[i].et, notes[i].lyric, notes[i].midi)
        i += 1
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
    args.threshold *= 1e-3 #args.threshold(default:30000 -> 30)
    #args.threshold *= 1e-2 #args.threshold(default:30000 -> 300)
    segments = []
    
    #args.scp = data/tr_no_dev
    scorescp = open(os.path.join(args.scp, "score.scp"), "r", encoding="utf-8")
    update_segments = open(
        os.path.join(args.scp, "segments_from_xml.tmp"), "w", encoding="utf-8"
    )
    update_text = open(os.path.join(args.scp, "text.tmp"), "w", encoding="utf-8")
    reader = XMLReader(os.path.join(args.scp, "score.scp"))

    #xml_line = GTSINGER_Tenor1_Glissando_我的歌声里_0000 /data3/tyx/.../Control_Group/0000.musicxml
    for xml_line in scorescp:
        xmlline = xml_line.strip().split(" ")
        recording_id = xmlline[0] #GTSINGER_Tenor1_Glissando_我的歌声里_0000
        path = xmlline[1] #/data3/tyx/.../Control_Group/0000.musicxml
        
        #tempo = 143
        #temp_info = [<espnet2.fileio.score_scp.NOTE object at 0x7f25077913d0>,...]
        tempo, temp_info = reader[recording_id]
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

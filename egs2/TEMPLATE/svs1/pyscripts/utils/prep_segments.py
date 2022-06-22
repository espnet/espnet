#!/usr/bin/env python3
import argparse
import os
import sys


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
    parser.add_argument("win_size", type=int, help="window size in ms")
    parser.add_argument("win_shift", type=int, help="window shift in ms")
    return parser


def same_split(alignment, threshold):
    size = 2
    while (alignment[-1][1] - alignment[0][0]) / size > threshold:
        size += 1
    segments = []
    start = 0
    for i in range(size - 2):
        index = start
        while (
            index + 1 < len(alignment)
            and alignment[index + 1][1] - alignment[start][0] <= threshold
        ):
            index += 1
        segments.append(alignment[start : index + 1])
        start = index + 1
    segments.append(alignment[start:])
    return segments, size


def make_segment(file_id, alignment, threshold=13500 * 1e-3, sil="pau"):
    segment_info = {}
    start_id = 1
    seg_start = []
    seg_end = []
    for i in range(len(alignment)):
        if len(seg_start) == len(seg_end) and sil not in alignment[i][2]:
            seg_start.append(i)
        elif len(seg_start) != len(seg_end) and sil in alignment[i][2]:
            seg_end.append(i)
        else:
            continue
    if len(seg_start) != len(seg_end):
        seg_end.append(len(alignment) - 1)
    if len(seg_start) <= 1:
        start = alignment[seg_start[0]][0]
        end = alignment[seg_end[0]][0]

        st, ed = seg_start[0], seg_end[0]
        if end - start > threshold:
            segments, size = same_split(alignment[st:ed], threshold)
            for i in range(size):
                segment_info[pack_zero(file_id, start_id)] = segments[i]
                start_id += 1
        else:
            segment_info[pack_zero(file_id, start_id)] = alignment[st:ed]

    else:
        for i in range(len(seg_start)):
            start = alignment[seg_start[i]][0]
            end = alignment[seg_end[i]][0]
            st, ed = seg_start[i], seg_end[i]
            if end - start > threshold:
                segments, size = same_split(alignment[st:ed], threshold)
                for i in range(size):
                    segment_info[pack_zero(file_id, start_id)] = segments[i]
                    start_id += 1
                continue

            segment_info[pack_zero(file_id, start_id)] = alignment[st:ed]
            start_id += 1
    return segment_info


if __name__ == "__main__":
    # print(sys.path[0]+'/..')
    os.chdir(sys.path[0] + "/..")
    # print(os.getcwd())
    args = get_parser().parse_args(sys.argv[1:])
    args.threshold *= 1e-3
    segments = []

    with open(args.scp + "/wav.scp") as f:
        for line in f:
            if len(line) == 0:
                continue
            recording_id, path = line.replace("\n", "").split(" ")
            lab_path = path.replace("wav/", "mono_label/").replace(".wav", ".lab")
            assert os.path.exists(lab_path)
            with open(lab_path) as lab_f:
                labels = []
                for lin in lab_f:
                    label = lin.replace("\n", "").split(" ")
                    if len(label) != 3:
                        continue
                    labels.append([float(label[0]), float(label[1]), label[2]])
                segments.append(make_segment(recording_id, labels, args.threshold))

    for file in segments:
        for key, val in file.items():
            segment_begin = "{:.7f}".format(val[0][0])
            segment_end = "{:.7f}".format(val[-1][1])

            sys.stdout.write("{} {} {}\n".format(key, segment_begin, segment_end))
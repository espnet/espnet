import argparse
import os
import shutil
from espnet2.fileio.read_text import read_label


def write_file(reader, writer, utt_map):
    for line in reader:
        uid = line.strip().split(" ")[0]
        if uid in utt_map:
            continue
        writer.write(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check Data Legality")
    parser.add_argument("src_data", type=str, help="source data directory")
    args = parser.parse_args()

    labels = read_label(os.path.join(args.src_data, "label"))
    utt_key = {}
    for key in labels:
        label_info = labels[key]
        skip_tag = False
        for st, et, phn in label_info:
            st = float(st)
            et = float(et)
            if st > et:
                skip_tag = True
                break
        if skip_tag:
            utt_key[key] = True
    wavscp = open(os.path.join(args.src_data, "wav.scp"), "r", encoding="utf-8")
    scorescp = open(os.path.join(args.src_data, "score.scp"), "r", encoding="utf-8")
    utt2spk = open(os.path.join(args.src_data, "utt2spk"), "r", encoding="utf-8")
    label_scp = open(os.path.join(args.src_data, "label"), "r", encoding="utf-8")
    text = open(os.path.join(args.src_data, "text"), "r", encoding="utf-8")

    label_writer = open(os.path.join(args.src_data, "label.tmp"), "w", encoding="utf-8")
    score_writer = open(os.path.join(args.src_data, "score.scp.tmp"), "w", encoding="utf-8")
    utt2spk_writer = open(os.path.join(args.src_data, "utt2spk.tmp"), "w", encoding="utf-8")
    wavscp_writer = open(os.path.join(args.src_data, "wav.scp.tmp"), "w", encoding="utf-8")
    text_writer = open(os.path.join(args.src_data, "text.tmp"), "w", encoding="utf-8")

    write_file(wavscp, wavscp_writer, utt_key)
    write_file(scorescp, score_writer, utt_key)
    write_file(utt2spk, utt2spk_writer, utt_key)
    write_file(label_scp, label_writer, utt_key)
    write_file(text, text_writer, utt_key)
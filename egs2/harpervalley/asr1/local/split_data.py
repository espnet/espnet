import argparse
import os
import random


def get_spk_list(utt2spk_path, min_spk_utt=10, train_frac=0.8, val_frac=0.1):
    # collect spk_id and count
    with open(utt2spk_path, "r", encoding="utf-8") as utt2spk:
        spk_dict = {}
        for line in utt2spk.readlines():
            spk_id = line.strip().split()[1]
            if spk_id in spk_dict:
                spk_dict[spk_id] += 1
            else:
                spk_dict[spk_id] = 1

    # create spk_list
    spk_list = []
    for spk_id, count in spk_dict.items():
        if count > min_spk_utt:
            spk_list.append(spk_id)

    # split spk_list
    spk_list = sorted(spk_list)
    num_speaker = len(spk_list)
    num_train = int(train_frac * num_speaker)
    num_val = int(val_frac * num_speaker)
    random.seed(42)
    random.shuffle(spk_list)

    train_spk_list = spk_list[:num_train]
    dev_spk_list = spk_list[num_train : num_train + num_val]
    test_spk_list = spk_list[num_train + num_val :]

    return train_spk_list, dev_spk_list, test_spk_list


def split_files(source_dir, min_spk_utt, train_frac, val_frac):
    wavscp_train = open("data/train/wav.scp", "w", encoding="utf-8")
    utt2spk_train = open("data/train/utt2spk", "w", encoding="utf-8")
    segments_train = open("data/train/segments", "w", encoding="utf-8")
    text_train = open("data/train/text", "w", encoding="utf-8")
    wavscp_dev = open("data/valid/wav.scp", "w", encoding="utf-8")
    utt2spk_dev = open("data/valid/utt2spk", "w", encoding="utf-8")
    segments_dev = open("data/valid/segments", "w", encoding="utf-8")
    text_dev = open("data/valid/text", "w", encoding="utf-8")
    wavscp_test = open("data/test/wav.scp", "w", encoding="utf-8")
    utt2spk_test = open("data/test/utt2spk", "w", encoding="utf-8")
    segments_test = open("data/test/segments", "w", encoding="utf-8")
    text_test = open("data/test/text", "w", encoding="utf-8")

    train_spk_list, dev_spk_list, test_spk_list = get_spk_list(
        os.path.join(source_dir, "utt2spk"), min_spk_utt, train_frac, val_frac
    )

    # split wav.scp
    with open(os.path.join(source_dir, "wav.scp"), "r", encoding="utf-8") as wavscp:
        for line in wavscp.readlines():
            rec_id = line.strip().split()[0]
            spk_id = rec_id.split("-")[0]
            if spk_id in train_spk_list:
                wavscp_train.write("{}\n".format(line.strip()))
            elif spk_id in dev_spk_list:
                wavscp_dev.write("{}\n".format(line.strip()))
            elif spk_id in test_spk_list:
                wavscp_test.write("{}\n".format(line.strip()))

    # split utt2spk
    with open(os.path.join(source_dir, "utt2spk"), "r", encoding="utf-8") as utt2spk:
        for line in utt2spk.readlines():
            spk_id = line.strip().split()[1]
            if spk_id in train_spk_list:
                utt2spk_train.write("{}\n".format(line.strip()))
            elif spk_id in dev_spk_list:
                utt2spk_dev.write("{}\n".format(line.strip()))
            elif spk_id in test_spk_list:
                utt2spk_test.write("{}\n".format(line.strip()))

    # split segments
    with open(os.path.join(source_dir, "segments"), "r", encoding="utf-8") as segments:
        for line in segments.readlines():
            utt_id = line.strip().split()[0]
            spk_id = utt_id.split("-")[0]
            if spk_id in train_spk_list:
                segments_train.write("{}\n".format(line.strip()))
            elif spk_id in dev_spk_list:
                segments_dev.write("{}\n".format(line.strip()))
            elif spk_id in test_spk_list:
                segments_test.write("{}\n".format(line.strip()))

    # split text
    with open(os.path.join(source_dir, "text"), "r", encoding="utf-8") as text:
        for line in text.readlines():
            utt_id = line.strip().split()[0]
            spk_id = utt_id.split("-")[0]
            if spk_id in train_spk_list:
                text_train.write("{}\n".format(line.strip()))
            elif spk_id in dev_spk_list:
                text_dev.write("{}\n".format(line.strip()))
            elif spk_id in test_spk_list:
                text_test.write("{}\n".format(line.strip()))

    wavscp_train.close()
    utt2spk_train.close()
    segments_train.close()
    text_train.close()
    wavscp_dev.close()
    utt2spk_dev.close()
    segments_dev.close()
    text_dev.close()
    wavscp_test.close()
    utt2spk_test.close()
    segments_test.close()
    text_test.close()


parser = argparse.ArgumentParser()
parser.add_argument("--source_dir", type=str, default="data/tmp")
parser.add_argument("--min_spk_utt", type=int, default=10)
parser.add_argument("--train_frac", type=float, default=0.8)
parser.add_argument("--val_frac", type=float, default=0.1)

args = parser.parse_args()

split_files(args.source_dir, args.min_spk_utt, args.train_frac, args.val_frac)

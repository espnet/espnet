"""
Split AphasiaBank into train, test and val sets, according to config.py
"""
import os
from argparse import ArgumentParser

import numpy as np
from data import test_spks, train_spks, utt2spk, val_spks


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--text", type=str, help="Path to text", required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    return parser.parse_args()


spk_splits = [train_spks, val_spks, test_spks]


def main():
    args = get_args()

    # get all speakers
    spk2utts = {}
    utt2trans = {}
    with open(args.text, encoding="utf-8") as f:
        for line in f:
            utt, trans = line.rstrip("\n").split(maxsplit=1)
            spk = utt2spk(utt)
            spk2utts.setdefault(spk, []).append(utt)
            utt2trans[utt] = trans

    splits = ["train", "val", "test"]
    out_dir = args.out_dir

    # print percentage of speakers in each split
    split_percent = np.asarray(
        [len(train_spks), len(val_spks), len(test_spks)], dtype=float
    )
    split_percent /= np.sum(split_percent)
    print(f"Percentage of train, val and test speakers (PWA): {split_percent}")

    for i, s in enumerate(splits):
        subset_dir = os.path.join(out_dir, s)
        os.makedirs(subset_dir, exist_ok=True)

        utt_list = open(os.path.join(subset_dir, "utt.list"), "w", encoding="utf-8")
        text = open(os.path.join(subset_dir, "text"), "w", encoding="utf-8")
        utt2spk_file = open(os.path.join(subset_dir, "utt2spk"), "w", encoding="utf-8")

        spks = spk_splits[i]
        for spk in spks:
            if spk not in spk2utts:
                print(
                    f"Skipping utterances of {spk} "
                    f"since they are not found in {args.text}"
                )
                continue

            for utt in spk2utts[spk]:
                utt_list.write(f"{utt}\n")
                text.write(f"{utt}\t{utt2trans[utt]}\n")
                utt2spk_file.write(f"{utt}\t{spk}\n")

        utt_list.close()
        text.close()
        utt2spk_file.close()


if __name__ == "__main__":
    main()

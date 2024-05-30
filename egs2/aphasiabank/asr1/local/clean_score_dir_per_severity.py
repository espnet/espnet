"""
Filter out *.trn files in score_cer and score_wer based on language and
aph types
"""

import argparse
import os
from typing import Iterable

from data import severity2spks, utt2spk


def filter_text(in_file: str, out_file: str, spks: Iterable[str]):
    with open(in_file, encoding="utf-8") as f, open(
        out_file, "w", encoding="utf-8"
    ) as of:
        for line in f:
            utt = line.split()[-1].replace("(", "").replace(")", "")
            spk = utt2spk(utt)
            if spk in spks:
                of.write(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("in_dir", type=str)
    parser.add_argument("out_dir", type=str)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for file in ["hyp", "ref"]:
        for sev, spks in severity2spks.items():
            filter_text(
                os.path.join(args.in_dir, f"{file}.trn"),
                os.path.join(args.out_dir, f"{file}.{sev}.trn"),
                spks,
            )


if __name__ == "__main__":
    main()

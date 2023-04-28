"""
Adapted from egs2/fleurs/asr1/local/create_lids.py

Create a file with the aphasia label for each utterance, used by InterCTC-based
Aphasia detection models
"""

import argparse

from data import spk2aphasia_label, utt2spk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str)
    args = parser.parse_args()

    with open(f"{args.data_dir}/text", encoding="utf-8") as in_file, open(
        f"{args.data_dir}/utt2aph", "w", encoding="utf-8"
    ) as utt_file:
        for line in in_file:
            line = line.rstrip("\n")

            utt_id = line.split()[0]
            spk = utt2spk(utt_id)

            aph = spk2aphasia_label[spk]
            aph = f"[{aph.upper()}]"

            utt_file.write(f"{utt_id} {aph}\n")


if __name__ == "__main__":
    main()

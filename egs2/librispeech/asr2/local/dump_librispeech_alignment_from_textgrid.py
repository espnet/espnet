# -*- coding: utf-8 -*-

import argparse
import glob
import os
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        usage="python local/dump_librispeech_alignment_from_textgrid.py "
        "--dataset 'dev-other' 'dev-clean'"
    )
    parser.add_argument(
        "--alignment_root", type=str, default="data/librispeech_phoneme_alignment"
    )
    parser.add_argument("--dataset", type=str, nargs="+", default=None)
    parser.add_argument("--downsample", type=int, default=1)

    args = parser.parse_args()

    assert args.downsample >= 1

    try:
        from praatio import textgrid
    except Exception as e:
        print("Error: praatio is not properly installed.")
        print("Please install praatio: . ./path.sh && python -m pip install praatio")
        raise e

    for dset in args.dataset:
        #  data/librispeech_phoneme_alignment/dev-clean/1272/128104/1272-128104-00
        with open(f"{args.alignment_root}/{dset}.tsv", "w") as fd:
            for f in glob.glob(f"{args.alignment_root}/{dset}/*/*/*", recursive=True):
                uid = os.path.splitext(os.path.basename(f))[0]
                tg = textgrid.openTextgrid(f, includeEmptyIntervals=True)
                phone_tier_list = tg.getTier("phones").entries

                align = []
                for item in phone_tier_list:
                    start_time = int(item.start * 100)
                    end_time = int(item.end * 100)
                    label = item.label
                    if label == "":
                        continue
                    align += [label for _ in range(end_time - start_time)]

                if args.downsample > 1:
                    align = align[:: args.downsample]

                out = f"{uid}\t" + ",".join(align)

                fd.write(out + "\n")

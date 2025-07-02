#!/usr/bin/env python3
# Data preparation for GLOBE-V2 ➜ Kaldi directories
# Author: your-name

import argparse
import os
from collections import defaultdict

import datasets  # 🤗 Datasets
from tqdm import tqdm
import soundfile as sf

# ----------------------------------------------------------- helpers
def write_kaldi_dir(split, items, outdir, fs_out):
    """items: list of (utt_id, spk_id, wav_path, txt, dur)"""
    os.makedirs(outdir, exist_ok=True)

    with open(f"{outdir}/wav.scp", "w") as fw, \
         open(f"{outdir}/text",   "w") as ft, \
         open(f"{outdir}/utt2spk", "w") as fu, \
         open(f"{outdir}/utt2dur", "w") as fd:

        spk2utt = defaultdict(list)

        for uid, spk, wav, txt, dur in items:
            # sox pipeline → 24 kHz wav on stdout
            fw.write(f"{uid} sox -t flac {wav} -r {fs_out} -t wav - |\n")
            ft.write(f"{uid} {txt}\n")
            fu.write(f"{uid} {spk}\n")
            fd.write(f"{uid} {dur:.3f}\n")
            spk2utt[spk].append(uid)

    with open(f"{outdir}/spk2utt", "w") as fs2u:
        for spk, utts in spk2utt.items():
            fs2u.write(f"{spk} {' '.join(utts)}\n")


# ----------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_set", required=True)
    parser.add_argument("--dev_set",   required=True)
    parser.add_argument("--test_set",  required=True)
    parser.add_argument("--dataset_path", required=True,
                        help="globe_v2 directory (contains parquet shards)")
    parser.add_argument("--dest_path", required=True,
                        help="output root for data/ dirs")
    parser.add_argument("--fs_out", type=int, default=24000,
                        help="sampling rate written to wav.scp (sox resample)")
    args = parser.parse_args()

    splits = {
        args.train_set: "train",
        args.dev_set:   "val",
        args.test_set:  "test",
    }

    for kaldi_name, hf_split in splits.items():
        print(f"Preparing split {hf_split} → data/{kaldi_name}")
        pattern = os.path.join(args.dataset_path, f"{hf_split}-*.parquet")
        # 🤗 datasets will glob the parquet shards automatically
        # ds = datasets.load_dataset(
        #     "parquet",
        #     data_files=pattern,
        #     split=hf_split,
        #     streaming=False,
        ds = datasets.load_dataset(
        "parquet",
        data_files=f"{args.dataset_path}/data/{hf_split}-*.parquet",
        split=hf_split,
        streaming=False,
        cache_dir="/ocean/projects/cis210027p/ttao3/cache",
    )
        # avoid decoding audio; keep only metadata
        ds = ds.cast_column("audio", datasets.Audio(decode=False))

        items = []
        for row in tqdm(ds, desc=hf_split):
            wav  = row["audio"]["path"]
            spk  = row["speaker_id"]
            txt  = row["transcript"].strip()
            # use the HF-provided column 'duration' instead of metadata
            dur  = float(row["duration"])
            utt  = f"{spk}_{os.path.basename(wav).split('.')[0]}"
            items.append((utt, spk, wav, txt, dur))


        outdir = os.path.join(args.dest_path, kaldi_name)
        write_kaldi_dir(kaldi_name, items, outdir, args.fs_out)

    print("✅  Data preparation done.")


if __name__ == "__main__":
    main()

import argparse
import os
from pathlib import Path

from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser(
        description="Process Raw data to Kaldi-like format"
    )
    parser.add_argument("--source", type=Path, default="downloads/mls_spanish")
    parser.add_argument("--lang", type=str, default="es")
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--target_dir", type=Path, default="data")
    return parser


def pad_zero(char, size):
    pad = "0" * (size - len(char))
    return pad + char


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    count = 0
    for subdir in ("train", "dev", "test"):
        target_dir = args.target_dir / f"{args.prefix}{args.lang}_{subdir}"
        if not target_dir.exists():
            target_dir.mkdir(parents=True)
        work_dir = args.source / subdir

        wavscp = open(target_dir / "wav.scp", "w", encoding="utf-8")
        text = open(target_dir / "text", "w", encoding="utf-8")
        utt2spk = open(target_dir / "utt2spk", "w", encoding="utf-8")
        spk2utt = open(target_dir / "spk2utt", "w", encoding="utf-8")
        spk_save = {}

        for line in tqdm(open(work_dir / "transcripts.txt")):
            org_header, trans = line.split("\t")
            org_spk, book, seg = org_header.split("_")
            spk = pad_zero(org_spk, 6)
            header = f"{spk}_{book}_{seg}"
            wavdir = work_dir / org_spk / book / f"{org_header}.flac"
            wavscp.write(f"{header} flac -c -d --silent {wavdir} |\n")
            text.write(f"{header} {trans}")
            utt2spk.write(f"{header} {spk}\n")
            spk_save[spk] = spk_save.get(spk, []) + [header]

        for spk in spk_save.keys():
            utts = " ".join(spk_save[spk])
            spk2utt.write(f"{spk} {utts}\n")

    print("pre-processing finished")

import argparse
import os
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser(
        description="Process Raw data to Kaldi-like format"
    )
    parser.add_argument("--source", type=Path, default=Path("downloads/mls_polish"))
    parser.add_argument(
        "--lang",
        type=str,
        default="pl",
        choices=["es", "en", "fr", "nl", "it", "pt", "pl", "de"],
    )
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--target_dir", type=Path, default=Path("data"))
    parser.add_argument("--data_split", default="full", choices=["full", "10h", "1h"])
    return parser


def pad_zero(char, size):
    pad = "0" * (size - len(char))
    return pad + char


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    for subdir in ("train", "dev", "test"):
        target_dir = args.target_dir / f"{args.prefix}{args.lang}_{subdir}"
        if not target_dir.exists():
            target_dir.mkdir(parents=True)

        if subdir == "train":
            subsamples = set()
            handle_root = args.source / subdir / "limited_supervision"
            if args.data_split == "1h" or args.data_split == "10h":
                for one_hr_split in range(6):
                    for line in open(
                        handle_root / "1hr" / str(one_hr_split) / "handles.txt"
                    ):
                        subsamples.add(line.strip())
            if args.data_split == "10h":
                for line in open(handle_root / "9hr" / "handles.txt"):
                    subsamples.add(line.strip())
        else:
            subsamples = set()

        wavscp = open(target_dir / "wav.scp", "w", encoding="utf-8")
        text = open(target_dir / "text", "w", encoding="utf-8")
        utt2spk = open(target_dir / "utt2spk", "w", encoding="utf-8")
        spk2utt = open(target_dir / "spk2utt", "w", encoding="utf-8")
        spk_save = defaultdict(list)

        count = 0
        for line in tqdm(open(args.source / subdir / "transcripts.txt")):
            org_header, trans = line.strip().split("\t")
            if subsamples and org_header not in subsamples:
                continue

            org_spk, book, seg = org_header.split("_")
            spk = pad_zero(org_spk, 6)
            header = f"{spk}_{book}_{seg}"
            wavdir = (
                args.source / subdir / "audio" / org_spk / book / f"{org_header}.flac"
            )
            wavscp.write(f"{header} flac -c -d --silent {wavdir} |\n")
            text.write(f"{header} {trans}\n")
            utt2spk.write(f"{header} {spk}\n")
            spk_save[spk].append(header)

            count += 1

        for spk in spk_save.keys():
            utts = " ".join(spk_save[spk])
            spk2utt.write(f"{spk} {utts}\n")

        print(f"{subdir}: {count} lines written.")

    print("pre-processing finished.")

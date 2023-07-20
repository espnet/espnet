import argparse
import os
import sys

import soundfile as sf

# from tqdm import tqdm
from tqdm.contrib import tzip


def main(args):
    with open(args.scp, "r") as f:
        lines_scp = f.readlines()
    with open(args.utt2spk, "r") as f:
        lines_u2s = f.readlines()

    with open(args.scp + "2", "w") as f_scp, open(args.utt2spk + "2", "w") as f_u2s:
        assert len(lines_scp) == len(lines_u2s)

        for scp, u2s in tzip(lines_scp, lines_u2s):
            try:
                audio = sf.read(scp.strip().split(" ")[1])
                f_scp.write(scp)
                f_u2s.write(u2s)
            except sf.LibsndfileError:
                continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trial mapper")
    parser.add_argument(
        "--scp",
        type=str,
        required=True,
        help="directory of wav.scp file",
    )
    parser.add_argument(
        "--utt2spk",
        type=str,
        required=True,
        help="directory of wav.scp file",
    )
    args = parser.parse_args()

    sys.exit(main(args))

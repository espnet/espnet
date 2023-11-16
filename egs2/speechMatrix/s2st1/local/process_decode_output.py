"""
Convert a .npy file containing decoded discrete units into text that contains a sequence of integers per line
"""
import argparse
import logging
import os
import numpy as np

logging.basicConfig()
logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    args = get_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "utt2units")
    logger.info(f"Writing to {out_path}")

    with open(args.feats_scp, encoding='utf-8') as f, open(out_path, "w", encoding='utf-8') as of:
        for line in f:
            utt, path = line.rstrip('\n').split()

            data = np.load(path)
            assert len(data.shape) == 1
            # minus 2 since "0" has index of 2, see also the token list under data/
            data = [str(e - 2) for e in data]
            data = data[:args.max_len]
            s = ' '.join(data)

            of.write(f'{utt}\t{s}\n')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feats_scp", type=str, required=True, help=".scp file for discrete unit features"
    )
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--max_len", type=int, default=400)
    return parser.parse_args()


if __name__ == "__main__":
    main()

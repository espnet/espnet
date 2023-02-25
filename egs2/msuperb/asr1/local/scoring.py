import argparse
import glob
import json
import os

from tqdm import tqdm


def lid(args):
    print("LID Scoring...")
    pass


def error_rate(args):
    print("WER/CER Scoring...")
    roots = []
    for paths in glob.glob(f"{args.dir}/*/*/*/independent/*/ref.trn"):
        roots.append(os.path.dirname(paths))
    for paths in glob.glob(f"{args.dir}/*/*/*/few_shot/*/ref.trn"):
        roots.append(os.path.dirname(paths))
    for paths in glob.glob(f"{args.dir}/*/*/*/language_family/*/ref.trn"):
        roots.append(os.path.dirname(paths))

    # Call sclite API for each directory!
    for root in tqdm(roots):
        os.system(
            f"sclite -r {root}/ref.trn trn -h {root}/hyp.trn trn -i rm -o all stdout > {root}/result.txt"
        )


def main(args):
    error_rate(args)
    # lid(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str)

    args = parser.parse_args()
    main(args)

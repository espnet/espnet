"""Filter repeated English ASR utterances."""

from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


def parse_args():
    parser = ArgumentParser(
        description="Rename uttid and wavid to filter out repeated utterances"
    )
    parser.add_argument("--input", type=Path, help="Input data directory.")
    parser.add_argument("--output", type=Path, help="Output data directory.")
    args = parser.parse_args()
    return args


def rename(input: str) -> str:
    input = input.split("_")
    del input[6]
    del input[1]
    return "_".join(input)


def main(input: Path, output: Path):
    with open(input / "segments", "r") as fin, open(output / "segments", "w") as fout:
        for line in fin:
            line = line.strip().split()
            line = [rename(line[0]), rename(line[1]), line[2], line[3]]
            fout.write(" ".join(line) + "\n")

    for file in ["text", "text.ctc", "text.prev", "wav.scp"]:
        with open(input / file, "r") as fin, open(output / file, "w") as fout:
            for line in fin:
                line = line.strip().split(maxsplit=1)
                line = [rename(line[0]), line[1]]
                fout.write(" ".join(line) + "\n")

    with open(input / "utt2spk", "r") as fin, open(output / "utt2spk", "w") as fout:
        for line in fin:
            line = line.strip().split()
            line = [rename(line[0]), rename(line[1])]
            fout.write(" ".join(line) + "\n")


if __name__ == "__main__":
    args = parse_args()
    main(input=args.input, output=args.output)

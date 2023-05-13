from argparse import ArgumentParser
from pathlib import Path


def parse_args():
    parser = ArgumentParser(description="Remove invalid whitespace from text.")
    parser.add_argument("--input", type=Path, help="Input file.")
    parser.add_argument("--output", type=Path, help="Output file.")

    args = parser.parse_args()
    return args


def process_file(input: Path, output: Path):
    with input.open("r") as fin, output.open("w") as fout:
        for line in fin:
            line = line.strip()
            line = " ".join(line.split())
            fout.write(line + "\n")


if __name__ == "__main__":
    args = parse_args()
    process_file(input=args.input, output=args.output)

"""
Remove [APH] and [NONAPH] tags from the hypothesis file.
Works for both character- and word-level tokenization.
"""
from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("output", type=str)
    return parser.parse_args()


def clean_line(line: str) -> str:
    line, utt = line.rsplit("(", maxsplit=1)
    line: str = (
        line.replace("[APH]", "")
        .replace("[NONAPH]", "")
        .replace("[ A P H ]", "")
        .replace("[ N O N A P H ]", "")
    )
    line = line.strip().removeprefix("<space>").removesuffix("<space>")
    return f"{line}\t({utt}"


def main():
    args = get_args()

    with open(args.input, encoding="utf-8") as f, open(
        args.output, "w", encoding="utf-8"
    ) as of:
        for line in f:
            of.write(clean_line(line))


if __name__ == "__main__":
    main()

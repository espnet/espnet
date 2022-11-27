#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path

import numpy as np
from espnet.utils.cli_utils import get_commandline_args


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Insert silence token",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--word_boundary", type=str, required=True)
    parser.add_argument("--sil_token", type=str, default="<SIL>")
    parser.add_argument("--sil_prob", type=float, required=True)
    parser.add_argument("--input", "-i", required=True, help="Input text.")
    parser.add_argument("--output", "-o", required=True, help="Output text.")
    return parser


def insert_silence(
    word_boundary: str,
    sil_token: str,
    sil_prob: float,
    input: str,
    output: str,
):
    assert sil_prob >= 0.0 and sil_prob <= 1.0

    input_f = Path(input).open("r", encoding="utf-8")
    output_p = Path(output)
    output_p.parent.mkdir(parents=True, exist_ok=True)
    output_f = output_p.open("w", encoding="utf-8")

    for line in input_f:
        line = line.rstrip()
        line_list = line.split(word_boundary)
        line_length = len(line_list)

        output_list = []
        if line_length > 1:
            sample_sil_probs = np.random.random(line_length - 1)

            for index in range(line_length - 1):
                phones = line_list[index]
                output_list.append(phones)
                if sil_prob >= sample_sil_probs[index]:
                    output_list.append(sil_token)
            output_list.append(line_list[-1])

            output = " ".join(output_list)
            output_f.write(f"{output}\n")
        else:
            output_f.write(f"{line}\n")

    output_f.close()


def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)

    insert_silence(**kwargs)


if __name__ == "__main__":
    main()

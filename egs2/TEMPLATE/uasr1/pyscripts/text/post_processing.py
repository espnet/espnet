#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

from espnet2.utils.types import str2bool
from espnet.utils.cli_utils import get_commandline_args


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Post-process text file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--word_boundary", type=str, required=True)
    parser.add_argument("--sil_token", type=str, default="<SIL>")
    parser.add_argument("--sil_prob", type=float, default=0.5, required=True)
    parser.add_argument("--input_text", required=True, help="Input text.")
    parser.add_argument("--output_text", required=True, help="Output text.")
    parser.add_argument("--output_vocab", help="Output vocabulary.")
    parser.add_argument(
        "--reduce_vocab", type=str2bool, default=True, help="reduce vocabulary"
    )
    return parser


def insert_silence(
    line_list: List,
    sil_prob: float,
    sil_token: str,
):
    line_length = len(line_list)

    list_with_silence = []
    if line_length > 1:
        sample_sil_probs = np.random.random(line_length - 1)

        for index in range(line_length - 1):
            phones = line_list[index]
            list_with_silence.append(phones)
            if sil_prob >= sample_sil_probs[index]:
                list_with_silence.append(sil_token)
        list_with_silence.append(line_list[-1])

    return list_with_silence


def filter_line(
    line_list: List,
    bad_word_set: List,
):
    filter_line_list = [word for word in line_list if word not in bad_word_set]
    return filter_line_list


def remove_digits(line_list: List):
    line_no_digits = []
    for word in line_list:
        word_no_digits = "".join(char for char in word if not char.isdigit())
        line_no_digits.append(word_no_digits)
    return line_no_digits


def post_processing(
    word_boundary: str,
    sil_token: str,
    sil_prob: float,
    input_text: str,
    output_text: str,
    output_vocab: Optional[str],
    reduce_vocab: str2bool = False,
):
    assert sil_prob >= 0.0 and sil_prob <= 1.0
    vocab_set = set()

    input_f = Path(input_text).open("r", encoding="utf-8")

    output_p = Path(output_text)
    output_p.parent.mkdir(parents=True, exist_ok=True)
    output_f = output_p.open("w", encoding="utf-8")

    for line in input_f:
        line = line.rstrip()
        line_list = line.split(word_boundary)

        if reduce_vocab:
            bad_word_set = set(r"'")
            line_list = filter_line(line_list, bad_word_set)
            line_list = remove_digits(line_list)

        if sil_prob > 0.0:
            line_list = insert_silence(line_list, sil_prob, sil_token)

        output_f.write(f"{' '.join(line_list)}\n")

        for token in line_list:
            vocab_set.add(token)

    if output_vocab:
        output_vp = Path(output_vocab)
        output_vp.parent.mkdir(parents=True, exist_ok=True)
        output_v = output_vp.open("w", encoding="utf-8")

        for token in vocab_set:
            output_v.write(f"{token}\n")


def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)

    post_processing(**kwargs)


if __name__ == "__main__":
    main()

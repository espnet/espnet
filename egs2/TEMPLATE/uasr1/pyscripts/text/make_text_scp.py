#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path
from mmap import mmap
from espnet.utils.cli_utils import get_commandline_args


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Insert silence token",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input_text", type=str, required=True)
    parser.add_argument("--output_scp", type=str, required=True)
    parser.add_argument("--num_digits", type=int, default=10)
    return parser


def convert_digits(
    input_number: int,
    num_digits: int,
):
    input_number_digits = len(str(input_number))
    assert input_number_digits <= num_digits
    output_digits = (num_digits - input_number_digits) * "0" + str(input_number)

    return output_digits


def make_text_scp(
    input_text: str,
    output_scp: str,
    num_digits: int,
):
    input_text_path = Path(input_text)
    input_text_file = input_text_path.open("r+b")
    input_text_mm = mmap(input_text_file.fileno(), 0)
    output_scp_path = Path(output_scp)
    output_scp_file = output_scp_path.open("w")

    output_scp_file.write(f"{num_digits}\n")
    start_num_bytes = 0
    while True:
        line = input_text_mm.readline()
        if not line:
            break
        line_length = len(line) - 1
        end_num_bytes = start_num_bytes + line_length

        start_digits = convert_digits(start_num_bytes, num_digits)
        end_digits = convert_digits(end_num_bytes, num_digits)

        output_scp_file.write(f"{start_digits}{end_digits}\n")
        start_num_bytes = end_num_bytes + 1

    input_text_mm.close()
    output_scp_file.close()


def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)

    make_text_scp(**kwargs)


if __name__ == "__main__":
    main()

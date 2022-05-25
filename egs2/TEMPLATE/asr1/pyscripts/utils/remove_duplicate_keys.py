#!/usr/bin/env python3

# Copyright 2021 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Remove duplicate index from a index file."""

import argparse
from pathlib import Path


def read_2column_text(path):
    """Read a text file having 2 column as dict object.
    Examples:
        wav.scp:
            key1 /some/path/a.wav
            key2 /some/path/b.wav
        >>> read_2column_text('wav.scp')
        {'key1': '/some/path/a.wav', 'key2': '/some/path/b.wav'}
    """

    keys = set()
    with Path(path).open("r", encoding="utf-8") as f:
        for linenum, line in enumerate(f, 1):
            sps = line.rstrip().split(maxsplit=1)
            if len(sps) == 1:
                k, v = sps[0], ""
            else:
                k, v = sps
            if k in keys:
                continue
            else:
                print("{} {}".format(k, v))
                keys.add(k)


def main():
    """Print the duplicate-free result in stdout."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "index_file", type=str, help="Kaldi-style utterance-indexed file path."
    )
    args = parser.parse_args()

    read_2column_text(args.index_file)


if __name__ == "__main__":
    main()

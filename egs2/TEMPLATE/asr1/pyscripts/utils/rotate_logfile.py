#!/usr/bin/env python

# Copyright 2022 Chaitanya Narisetty
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Rotate log-file."""

import argparse
from pathlib import Path
import shutil


def rotate(path, max_num_log_files=1000):
    """Rotate a log-file while retaining past `max_num_log_files` files.
    Examples:
        /some/path/
        ├──logfile.txt
        ├──logfile.1.txt
        ├──logfile.2.txt
        >>> rotate('/some/path/logfile.txt')
        /some/path/
        ├──logfile.1.txt
        ├──logfile.2.txt
        ├──logfile.3.txt
    """
    for i in range(max_num_log_files - 1, -1, -1):
        if i == 0:
            p = Path(path)
            pn = p.parent / (p.stem + ".1" + p.suffix)
        else:
            _p = Path(path)
            p = _p.parent / (_p.stem + f".{i}" + _p.suffix)
            pn = _p.parent / (_p.stem + f".{i + 1}" + _p.suffix)

        if p.exists():
            if i == max_num_log_files - 1:
                p.unlink()
            else:
                shutil.move(p, pn)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "log_filepath", type=str, help="Path to log-file to be rotated."
    )
    parser.add_argument(
        "--max-num-log-files",
        type=int,
        help="Maximum number of log-files to be kept.",
        default=1000,
    )
    args = parser.parse_args()

    rotate(args.log_filepath, args.max_num_log_files)


if __name__ == "__main__":
    main()

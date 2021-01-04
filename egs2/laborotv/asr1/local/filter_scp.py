#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filters script file as in utils/filter_scp.pl.
The main difference from utils/fileter_scp.pl is that
if <id_list> has duplicated lines, this script prints the corresponding
lines in <input_scp> multiple times.
This script is for handing slightly buggy dataset.
"""

import argparse
from collections import Counter


def filter_scp(id_list_path, in_scp_path, field=1):
    """Filters script file as in utils/filter_scp.pl, but allows duplicates.

    Filters script file as in utils/filter_scp.pl.
    The main difference from utils/fileter_scp.pl is that
    if <id_list> has duplicated lines, this script prints the corresponding
    lines in <input_scp> multiple times.

    Args:
        id_list (str): path to id_list
        in_scp (str): path to input scp file

    Options:
        field (int): field to filter on
    """
    # load id_
    id_list = []
    with open(id_list_path, "r") as f:
        for line in f:
            id_list.append(line.rstrip().split()[0])
    id_list_count = Counter(id_list)

    # filter
    with open(in_scp_path, "r") as f:
        for line in f:
            curr_id = line.rstrip().split()[field - 1]
            if curr_id in id_list:
                for _ in range(id_list_count[curr_id]):
                    print(line.rstrip())


def main():
    """Main function"""
    parser = argparse.ArgumentParser()
    parser.add_argument("id_list", type=str)
    parser.add_argument("in_scp", type=str)
    parser.add_argument(
        "-f", "--field", type=int, default=1, help="field to filter on. (1, 2, 3...)"
    )
    args = parser.parse_args()

    filter_scp(id_list_path=args.id_list, in_scp_path=args.in_scp, field=args.field)


if __name__ == "__main__":
    main()

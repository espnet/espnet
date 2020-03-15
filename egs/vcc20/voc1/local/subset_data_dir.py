#!/usr/bin/env python3
# encoding: utf-8

# This script creates a subset of data, consisting of some specified number of utterances.

import argparse
from io import open
import logging
import sys

def get_parser():
    parser = argparse.ArgumentParser(
        description='creates a subset of data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--utt_list', type=str,
                        help='utt list file')
    parser.add_argument('--scp', type=str,
                        help='scp file')
    parser.add_argument('--verbose', '-V', default=1, type=int,
                        help='Verbose option')
    parser.add_argument('--out', '-O', type=str,
                        help='The output filename. '
                             'If omitted, then output to sys.stdout')
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    with open(args.utt_list, 'r') as f:
        utts = [line.rsplit()[0] for line in f.readlines()]
    with open(args.scp, 'r') as f:
        scps = f.readlines()

    if args.out is None:
        out = sys.stdout
    else:
        out = open(args.out, 'a', encoding='utf-8')
    
    for line in scps:
        number = line.split(' ')[0].split('_')[1]
        if number in utts:
            out.write(line)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
0: empty token for CTC algorithm
1: special UNK token
format: token + whitespace + index
"""
import argparse
import fileinput
from collections import defaultdict


def get_args():
    parser = argparse.ArgumentParser(description='generate vocabulary')
    parser.add_argument('--input', '-i', default=None, help='files to read, if empty, stdin is used')
    args = parser.parse_args()
    return args


def main(args):
    vocab = defaultdict(lambda: len(vocab) + 1)
    vocab['<unk>']
    for line in fileinput.input(args.input):
        tokens = line.strip().split()
        for token in tokens:
            vocab[token]
    vocab['<eos>']

    for key, value in sorted(vocab.items(), key=lambda x: x[1]):
        print('{} {}'.format(key, value))


if __name__ == "__main__":
    args = get_args()
    main(args)

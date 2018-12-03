#!/usr/bin/env python2

# Apache 2.0

import argparse
from builtins import str
from six import print_function

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exclude', '-v', dest='exclude', action='store_true', help='exclude filter words')
    parser.add_argument('filt', type=str, help='filter list')
    parser.add_argument('infile', type=str, help='input file')
    args = parser.parse_args()

    vocab = set()
    with open(args.filt) as vocabfile:
        for line in vocabfile:
            vocab.add(str(line, 'utf_8').strip())

    with open(args.infile) as textfile:
        for line in textfile:
            if args.exclude:
                print_function(" ".join(map(lambda word: word if word not in vocab else '', str(line, 'utf_8').strip()
                                            .split())).encode('utf_8'))
            else:
                print_function(" ".join(map(lambda word: word if word in vocab else '<UNK>', str(line, 'utf_8').strip()
                                            .split())).encode('utf_8'))

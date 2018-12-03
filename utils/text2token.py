#!/usr/bin/env python2

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import re
import sys

from builtins import str
from six import print_function


def exist_or_not(i, match_pos):
    start_pos = None
    end_pos = None
    for pos in match_pos:
        if pos[0] <= i < pos[1]:
            start_pos = pos[0]
            end_pos = pos[1]
            break

    return start_pos, end_pos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nchar', '-n', default=1, type=int,
                        help='number of characters to split, i.e., \
                        aabb -> a a b b with -n 1 and aa bb with -n 2')
    parser.add_argument('--skip-ncols', '-s', default=0, type=int,
                        help='skip first n columns')
    parser.add_argument('--space', default='<space>', type=str,
                        help='space symbol')
    parser.add_argument('--non-lang-syms', '-l', default=None, type=str,
                        help='list of non-linguistic symobles, e.g., <NOISE> etc.')
    parser.add_argument('text', type=str, default=False, nargs='?',
                        help='input text')
    args = parser.parse_args()

    rs = []
    if args.non_lang_syms is not None:
        with open(args.non_lang_syms, 'r') as f:
            nls = [str(x.rstrip(), 'utf_8') for x in f.readlines()]
            rs = [re.compile(re.escape(x)) for x in nls]

    if args.text:
        f = open(args.text)
    else:
        f = sys.stdin
    line = f.readline()
    n = args.nchar
    while line:
        x = str(line, 'utf_8').split()
        print_function(' '.join(x[:args.skip_ncols]).encode('utf_8')),
        a = ' '.join(x[args.skip_ncols:])

        # get all matched positions
        match_pos = []
        for r in rs:
            i = 0
            while i >= 0:
                m = r.search(a, i)
                if m:
                    match_pos.append([m.start(), m.end()])
                    i = m.end()
                else:
                    break

        if len(match_pos) > 0:
            chars = []
            i = 0
            while i < len(a):
                start_pos, end_pos = exist_or_not(i, match_pos)
                if start_pos is not None:
                    chars.append(a[start_pos:end_pos])
                    i = end_pos
                else:
                    chars.append(a[i])
                    i += 1
            a = chars

        a = [a[i:i + n] for i in range(0, len(a), n)]

        a_flat = []
        for z in a:
            a_flat.append("".join(z))

        a_chars = [z.replace(' ', args.space) for z in a_flat]
        print_function(' '.join(a_chars).encode('utf_8'))
        line = f.readline()


if __name__ == '__main__':
    main()

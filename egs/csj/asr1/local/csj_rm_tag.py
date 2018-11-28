#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import sys
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-ncols', '-s', default=0, type=int,
                        help='skip first n columns')
    parser.add_argument('text', type=str,
                        help='input text')
    args = parser.parse_args()

    if args.text:
        f = open(args.text)
    else:
        f = sys.stdin

    line = f.readline()
    while line:
        x = unicode(line, 'utf_8').split()
        print ' '.join(x[:args.skip_ncols]).encode('utf_8'),
        print ' '.join([str.split('+')[0] for str in x[args.skip_ncols:]]).encode('utf_8')
        line = f.readline()

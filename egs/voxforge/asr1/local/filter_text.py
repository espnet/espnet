#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import sys
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filter-list', '-f', type=str,
                        help='filter list')
    args = parser.parse_args()

    with open(args.filter_list) as f:
        fil = [unicode(x, 'utf_8').rstrip() for x in f.readlines()]

    for x in sys.stdin.readlines():
        # extract text parts
        text = ' '.join(unicode(x, 'utf_8').rstrip().split()[1:])
        if text in fil:
            print x.split()[0], text.encode('utf_8')
                                                

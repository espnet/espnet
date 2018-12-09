#!/usr/bin/env python2
# encoding: utf-8

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import codecs
import json
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--key', '-k', type=str,
                        help='key')
    args = parser.parse_args()

    new_line = {}
    stream_in = codecs.getreader("utf-8")(sys.stdin if sys.version_info[0] == 2 else sys.stdin.buffer)
    line = stream_in.readline()
    while line:
        x = line.rstrip().split()
        v = {args.key: ' '.join(x[1:])}
        new_line[x[0]] = v
        line = stream_in.readline()

    all_l = {'utts': new_line}

    # ensure "ensure_ascii=False", which is a bug
    jsonstring = json.dumps(all_l, indent=4, ensure_ascii=False, sort_keys=True, separators=(',', ': '))
    print(jsonstring)

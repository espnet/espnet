#!/usr/bin/env python3

from __future__ import print_function
from __future__ import unicode_literals

import argparse
import codecs
from io import open
import sys


PY2 = sys.version_info[0] == 2
sys.stdin = codecs.getreader('utf-8')(
    sys.stdin if PY2 else sys.stdin.buffer)
sys.stdout = codecs.getwriter('utf-8')(
    sys.stdout if PY2 else sys.stdout.buffer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--segments', required=True, type=str,
                        help='path to segments')
    parser.add_argument('--scp', required=True, type=str,
                        help='path to wav.scp')
    args = parser.parse_args()
    
    wav_scp = {}
    with open(args.scp, encoding='utf-8') as f:
        for line in f:
            x = line.split()
            wav_scp[x[0]] = ' '.join(x[1:])
    
    with open(args.segments, encoding='utf-8') as f:
        # A01F0019_0000136_0014199 A01F0019 0.136 14.199
        for line in f:
            x = line.split()
            t_s = float(x[2])
            t_e = float(x[3])
            print('{} {} sox -t wav - -t wav - trim {} {:.2f} |'.format(x[0], wav_scp[x[1]], t_s, t_e-t_s))

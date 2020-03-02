#!/usr/bin/env python3
# encoding: utf-8

import argparse
import codecs
from distutils.util import strtobool
from io import open
import json
import logging
import sys

# I wonder if I can successfully import this line...
from espnet.utils.cli_utils import get_commandline_args

def get_parser():
    parser = argparse.ArgumentParser(
        description='Make data.json of specified size.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--json_file', type=str,
                        help='Json file')
    parser.add_argument('--start', default=-1, type=int,
                        help='Start index')
    parser.add_argument('--end', default=-1, type=int,
                        help='End index')
    parser.add_argument('--verbose', '-V', default=1, type=int,
                        help='Verbose option')
    parser.add_argument('--out', '-O', type=str,
                        help='The output filename. '
                             'If omitted, then output to sys.stdout')
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    # logging info
    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    if args.verbose > 0:
        logging.basicConfig(level=logging.INFO, format=logfmt)
    else:
        logging.basicConfig(level=logging.WARN, format=logfmt)
    logging.info(get_commandline_args())

    with open(args.json_file, 'rb') as f:
        _json = json.load(f)['utts']

    data = {"utts" : {} }
    s, num = 0, 0
    for k, v in _json.items():
        
        if s < args.start:
            s += 1
            continue

        if num >= args.end - args.start:
            break
        
        data["utts"][k] = v
        num += 1

    print(len(data['utts']))
    
    if args.out is None:
        out = sys.stdout
    else:
        out = open(args.out, 'w', encoding='utf-8')
        
    json.dump(data, out,
               indent=4, ensure_ascii=False,
               separators=(',', ': '),
               )

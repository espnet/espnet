#!/usr/bin/env python3
# encoding: utf-8

import argparse
from io import open
import json
import logging
import sys

# I wonder if I can successfully import this line...
from espnet.utils.cli_utils import get_commandline_args


def get_parser():
    parser = argparse.ArgumentParser(
        description='Make json file for mel autoencoder training.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--json', type=str,
                        help='Json file')
    parser.add_argument('--num_utts', default=-1, type=int,
                        help='Number of utterances (take from head)')
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

    with open(args.json, 'rb') as f:
        _json = json.load(f)['utts']

    # get source and target speaker ()
    _ = list(_json.keys())[0].split('_')
    spk = _[0]
    spk = _[0]

    count = 0
    data = {"utts": {}}
    for k, v in _json.items():
        _ = k.split('_')
        number = '_'.join(_[1:])

        entry = {"input": _json[spk + '_' + number]['input'],
                 "output": _json[spk + '_' + number]['input'],
                 }
        data["utts"][number] = entry
        count += 1
        if args.num_utts > 0 and count >= args.num_utts:
            break

    if args.out is None:
        out = sys.stdout
    else:
        out = open(args.out, 'w', encoding='utf-8')

    json.dump(data, out,
              indent=4, ensure_ascii=False,
              separators=(',', ': '),
              )

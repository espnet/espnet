#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2020 Nagoya University (Wen-Chin Huang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
from io import open
import json
import logging
import sys

from espnet.utils.cli_utils import get_commandline_args


def get_parser():
    parser = argparse.ArgumentParser(
        description='Make json file for autoencoder-style pretraining.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input-json', type=str,
                        help='Json file for input')
    parser.add_argument('--output-json', type=str,
                        help='Json file for output')
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

    with open(args.input_json, 'rb') as f:
        input_json = json.load(f)['utts']
    with open(args.output_json, 'rb') as f:
        output_json = json.load(f)['utts']

    # Get source and target speaker (although they should be the same)
    _ = list(input_json.keys())[0].split('_')
    srcspk = _[0]
    _ = list(output_json.keys())[0].split('_')
    trgspk = _[0]

    count = 0
    data = {"utts": {}}
    # (dirty) loop through input only because in/out should have same files
    for k, v in input_json.items():
        _ = k.split('_')
        filename = '_'.join(_[1:])

        entry = {"input": input_json[srcspk + '_' + filename]['input'],
                 "output": output_json[trgspk + '_' + filename]['input'],
                 }
        entry["input"][0]['name'] = 'input1'
        entry["output"][0]['name'] = 'target1'
        data["utts"][k] = entry
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

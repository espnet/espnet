# -*- coding: utf-8 -*-
"""
extract recognized texts from data.<num>.json files
"""
import argparse
import glob
import json
import os
from itertools import takewhile


def get_args():
    parser = argparse.ArgumentParser(description='my script')
    parser.add_argument('--path', '-p', required=True, help='path to decode dir')
    args = parser.parse_args()
    return args


def process_json_file(path):
    data = json.load(open(path, 'r'))['utts']
    for idx, value in data.items():
        idx = idx.split('_')[-1]
        tokens = takewhile(lambda x: x != '<eos>', value['output'][0]['rec_token'].split(' '))
        txt = ' '.join(tokens)
        print('{}\t{}'.format(idx, txt))

    return None


def main(args):
    json_files = sorted(glob.glob(os.path.join(args.path, 'data.*.json')), key=lambda x: int(x.split('.')[-2]))
    for json_file in json_files:
        process_json_file(json_file)


if __name__ == "__main__":
    args = get_args()
    main(args)

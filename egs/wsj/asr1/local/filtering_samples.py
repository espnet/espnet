#!/usr/bin/env python3

# Copyright 2019 Shanghai Jiao Tong University (Wangyou Zhang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


import argparse
import json


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('json', type=str,
                        help='data.json')
    parser.add_argument('--min-io-ratio', type=float,
                        help='the minimum input-output length ratio for all samples', default=1.0)
    parser.add_argument('--min-io-delta', type=float,
                        help='an additional parameter for controlling the input-output length difference',
                        default=0.0)
    parser.add_argument('--output-json-path', type=str, required=True,
                        help='Output path of the filtered json file')
    args = parser.parse_args()

    # load dictionary
    with open(args.json, 'rb') as f:
        j = json.load(f)['utts']

    # remove samples with IO ratio smaller than args.min_io_ratio
    for key in list(j.keys()):
        ilen = j[key]['input'][0]['shape'][0]
        olen = min(x['shape'][0] for x in j[key]['output'])
        if float(ilen) - float(olen) * args.min_io_ratio < args.min_io_delta:
            j.pop(key)

    jsonstring = json.dumps({'utts': j}, indent=4, ensure_ascii=False, sort_keys=True)
    with open(args.output_json_path, 'w') as f:
        f.write(jsonstring)

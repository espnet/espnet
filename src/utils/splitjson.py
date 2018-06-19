#!/usr/bin/env python

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import json
import logging
import os

import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('json', type=str,
                        help='json files')
    parser.add_argument('-n', '--num-split', type=int, default=2,
                        help='number of splits')
    parser.add_argument('-o', '--outdir', type=str, default=None,
                        help='number of splits')
    args = parser.parse_args()

    # logging info
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")

    with open(args.json, 'r') as f:
        js = json.load(f)
    keys = js['utts'].keys()
    logging.info(args.json + ': has ' + str(len(keys)) + ' utterances')
    key_lists = np.array_split(keys, args.num_split)
    key_lists = [k.tolist() for k in key_lists]

    if args.outdir is not None:
        if not os.path.exists(args.outdir):
            os.makedirs(args.outdir)

    for idx, key_list in enumerate(key_lists, 1):
        split_js = {k: js['utts'][k] for k in key_list}
        if args.outdir is None:
            out = args.json.replace(".json", ".%d.json" % idx)
        else:
            out = args.outdir + "/" + os.path.basename(args.json).replace(".json", ".%d.json" % idx)
        logging.info(out + ': has ' + str(len(key_list)) + ' utterances')
        with open(out, 'w') as f:
            json.dump({'utts': split_js}, f,
                      indent=4,
                      sort_keys=True,
                      ensure_ascii=False)

#!/usr/bin/env python2
# encoding: utf-8

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
from __future__ import print_function
from __future__ import division

import argparse
import json
import logging
import os
import sys

import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('json', type=str,
                        help='json file')
    parser.add_argument('--parts', '-p', type=int,
                        help='Number of subparts to be prepared', default=0)
    args = parser.parse_args()

    # logging info
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")

    # check directory
    filename = os.path.basename(args.json).split('.')[0]
    dirname = os.path.dirname(args.json)
    dirname = '{}/split{}utt'.format(dirname, args.parts)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    # load json and split keys
    j = json.load(open(args.json))
    utt_ids = j['utts'].keys()
    if len(utt_ids) < args.parts:
        logging.error("#utterances < #splits. Use smaller split number.")
        sys.exit(1)
    utt_id_lists = np.array_split(utt_ids, args.parts)
    utt_id_lists = [utt_id_list.tolist() for utt_id_list in utt_id_lists]

    for i, utt_id_list in enumerate(utt_id_lists):
        new_dic = dict()
        for utt_id in utt_id_list:
            new_dic[utt_id] = j['utts'][utt_id]
        jsonstring = json.dumps({'utts': new_dic},
                                indent=4,
                                ensure_ascii=False,
                                sort_keys=True).encode('utf_8')
        fl = '{}/{}.{}.json'.format(dirname, filename, i + 1)
        sys.stdout = open(fl, "wb+")
        print(jsonstring)
        sys.stdout.close()

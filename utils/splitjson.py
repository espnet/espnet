#!/usr/bin/env python
# encoding: utf-8

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import codecs
import json
import logging
import os
import sys

import numpy as np

from espnet.utils.cli_utils import get_commandline_args


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('json', type=str,
                        help='json file')
    parser.add_argument('--parts', '-p', type=int,
                        help='Number of subparts to be prepared', default=0)
    args = parser.parse_args()

    # logging info
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    logging.info(get_commandline_args())

    # check directory
    filename = os.path.basename(args.json).split('.')[0]
    dirname = os.path.dirname(args.json)
    dirname = '{}/split{}utt'.format(dirname, args.parts)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    # load json and split keys
    j = json.load(codecs.open(args.json, 'r', encoding="utf-8"))
    utt_ids = sorted(list(j['utts'].keys()))
    logging.info("number of utterances = %d" % len(utt_ids))
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
                                sort_keys=True,
                                separators=(',', ': '))
        fl = '{}/{}.{}.json'.format(dirname, filename, i + 1)
        sys.stdout = codecs.open(fl, "w+", encoding="utf-8")
        print(jsonstring)
        sys.stdout.close()

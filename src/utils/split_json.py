#!/usr/bin/env python2
# encoding: utf-8

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
from __future__ import print_function

import argparse
import json
import logging
import os
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('json', type=str,
                        help='json file')
    parser.add_argument('--parts', '-p', type=int,
                        help='Number of subparts to be prepared', default=0)
    args = parser.parse_args()
    
    # logging info
    logging.basicConfig(level=logging.INFO, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")

    j = json.load(open(args.json))
    dics = [x for x in j['utts'].items()]

    ndics = len(dics)
    parts = [x for x in range (0, ndics+1, ndics // args.parts)]

    filename = os.path.basename(args.json).split('.')[0]
    dirname = os.path.dirname(args.json)
    dirname = '{}/split_nj{}'.format(dirname, args.parts)

    if not os.path.exists(dirname):
        os.makedirs(dirname)

    for i in range (args.parts):
        list_dic = dics[parts[i]:parts[i+1]]
        new_dic = {}
        for item in list_dic:
            id, dic = item
            new_dic[id] = dic
        jsonstring = json.dumps({'utts': new_dic}, indent=4, ensure_ascii=False).encode('utf_8')
        fl = '{}/{}.{}.json'.format(dirname, filename, i+1)

        sys.stdout = open(fl, "wb+")
        print(jsonstring)
        sys.stdout.close()

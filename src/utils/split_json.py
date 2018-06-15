#!/usr/bin/env python2
# encoding: utf-8

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import json
import argparse
import logging

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
    dics = j['utts']

    ndics = len(dics)
    parts = [x for x in range (0, ndics+1, ndics // args.parts]

    filename = os.path.basename(args.json).split('.')[0]
    for i in range (args.parts):
        new_dic = dics[parts[i]:parts[i+1]]
        jsonstring = json.dumps({'utts': new_dic}, indent=4, ensure_ascii=False).encode('utf_8')
        os.system('print {} > {}.{}.json'.format(jsonstring, filename, i))

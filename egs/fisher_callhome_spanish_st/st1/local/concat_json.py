#!/usr/bin/env python2
# encoding: utf-8

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import json
import logging


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('jsons', type=str, nargs='+',
                        help='json files')
    parser.add_argument('--verbose', '-V', default=0, type=int,
                        help='Verbose option')
    args = parser.parse_args()

    # logging info
    if args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    else:
        logging.basicConfig(
            level=logging.WARN, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")

    # make intersection set for utterance keys
    num_keys = 0
    new_dic = dict()
    for i, x in enumerate(args.jsons):
        with open(x, 'r') as f:
            j = json.load(f)
        ks = j['utts'].keys()
        logging.info(x + ': has ' + str(len(ks)) + ' utterances')
        num_keys += len(ks)
        if i > 0:
            for k in ks:
                new_dic[k + '.' + str(i)] = j['utts'][k]
        else:
            new_dic = j['utts']
    logging.info('new json has ' + str(num_keys) + ' utterances')

    # ensure "ensure_ascii=False", which is a bug
    jsonstring = json.dumps(
        {'utts': new_dic}, indent=4, ensure_ascii=False, sort_keys=True).encode('utf_8')
    print(jsonstring)

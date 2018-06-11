#!/usr/bin/env python2
# encoding: utf-8

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import json
import argparse
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
    js = []
    intersec_ks = []
    for x in args.jsons:
        with open(x, 'r') as f:
            j = json.load(f)
        ks = j['utts'].keys()
        logging.info(x + ': has ' + str(len(ks)) + ' utterances')
        if len(intersec_ks) > 0:
            intersec_ks = intersec_ks.intersection(set(ks))
        else:
            intersec_ks = set(ks)
        js.append(j)
    logging.info('new json has ' + str(len(intersec_ks)) + ' utterances')
        
    new_dic = {}
    for k in intersec_ks:
        v = js[0]['utts'][k]
        for j in js[1:]:
            v.update(j['utts'][k])
        new_dic[k] = v
        
    # ensure "ensure_ascii=False", which is a bug
    jsonstring = json.dumps({'utts': new_dic}, indent=4, ensure_ascii=False).encode('utf_8')
    print(jsonstring)

#!/usr/bin/env python2
# encoding: utf-8

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import print_function
from __future__ import unicode_literals

import argparse
import json
import logging
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('jsons', type=str, nargs='+',
                        help='json files')
    parser.add_argument('--multi', '-m', type=int,
                        help='Test the json file for multiple input/output', default=0)
    parser.add_argument('--verbose', '-V', default=0, type=int,
                        help='Verbose option')
    parser.add_argument('--output-json', default='', type=str,
                        help='output json file')
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
            if len(intersec_ks) == 0:
                logging.warning("No intersection")
                break
        else:
            intersec_ks = set(ks)
        js.append(j)
    logging.info('new json has ' + str(len(intersec_ks)) + ' utterances')

    old_dic = dict()
    for k in intersec_ks:
        v = js[0]['utts'][k]
        for j in js[1:]:
            v.update(j['utts'][k])
        old_dic[k] = v

    new_dic = dict()
    for key_id in old_dic:
        dic = old_dic[key_id]

        in_dic = {}
        # if dic.has_key(('idim', 'utf-8')):
        if 'idim' in dic:
            in_dic['shape'] = (int(dic['ilen']), int(dic['idim']))
        in_dic['name'] = 'input1'
        in_dic['feat'] = dic['feat']

        out_dic = {}
        out_dic['name'] = 'target1'
        out_dic['shape'] = (int(dic['olen']), int(dic['odim']))
        out_dic['text'] = dic['text']
        out_dic['token'] = dic['token']
        out_dic['tokenid'] = dic['tokenid']

        new_dic[key_id] = {'input': [in_dic], 'output': [out_dic],
                           'utt2spk': dic['utt2spk']}

    # ensure "ensure_ascii=False", which is a bug
    if args.output_json:
        sys.stdout = open(args.output_json, "w")
    print(json.dumps({'utts': new_dic}, indent=4, ensure_ascii=False, sort_keys=True, separators=(',', ': ')))

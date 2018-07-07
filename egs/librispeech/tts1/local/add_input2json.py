#!/usr/bin/env python2
# encoding: utf-8

# Copyright 2018 Nagoya University (Tomoki Hayashi)
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

    # updated original dict to keep intersection
    intersec_org_dic = dict()
    for k in intersec_ks:
        v = js[0]['utts'][k]
        intersec_org_dic[k] = v

    intersec_add_dic = dict()
    for k in intersec_ks:
        v = js[1]['utts'][k]
        for j in js[2:]:
            v.update(j['utts'][k])
        intersec_add_dic[k] = v

    new_dic = dict()
    for id in intersec_org_dic:
        orgdic = intersec_org_dic[id]
        adddic = intersec_add_dic[id]
        # original input
        in_org_dic = orgdic['input'][0]
        # additional input
        in_add_dic = {}
        if 'idim' in adddic and 'ilen' in adddic:
            in_add_dic['shape'] = [int(adddic['ilen']),
                                   int(adddic['idim'])]
        elif 'idim' in adddic:
            in_add_dic['shape'] = [int(adddic['idim'])]
        in_add_dic['name'] = 'speaker_embedding'
        in_add_dic['feat'] = adddic['feat']

        new_dic[id] = {'input': [in_org_dic, in_add_dic],
                       'output': orgdic['output'],
                       'utt2spk': orgdic['utt2spk']}

    # ensure "ensure_ascii=False", which is a bug
    jsonstring = json.dumps(
        {'utts': new_dic}, indent=4, ensure_ascii=False, sort_keys=True).encode('utf_8')
    print(jsonstring)

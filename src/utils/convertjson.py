#!/usr/bin/env python2
# encoding: utf-8

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import json
import argparse
import logging
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('in_json', type=str,
                        help='Filename of input json file')
    parser.add_argument('in_feat', type=str,
                        help=' Filename of input feature file (Kaldi scp)')
    args = parser.parse_args()
    
    # logging info
    logging.basicConfig(level=logging.INFO, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")

    with open(args.in_json, 'r') as f:
        j = json.load(f)
    old_dic = j['utts'.decode('utf-8')]

    scp_dic = {}
    with open(args.in_feat, 'r') as f:
        for line in f:
            id, ark = unicode(line, 'utf-8').split()
            scp_dic[id] = ark

    new_dic = {}
    for item in old_dic.items():
        id, dic = item
        scp = scp_dic[id]

        in_dic = {}
        if dic.has_key(unicode('idim', 'utf-8')):
            in_dic[unicode('shape', 'utf-8')] = (int(dic[unicode('ilen', 'utf-8')]), int(dic[unicode('idim', 'utf-8')]))
        in_dic[unicode('name', 'utf-8')] = unicode('input1', 'utf-8')
        in_dic[unicode('feat', 'utf-8')] = scp_dic[id]

        out_dic = {}
        out_dic[unicode('name', 'utf-8')] = unicode('target1', 'utf-8')
        out_dic[unicode('shape', 'utf-8')] = (int(dic[unicode('olen', 'utf-8')]), int(dic[unicode('odim', 'utf-8')]))
        out_dic[unicode('text', 'utf-8')] = dic[unicode('text', 'utf-8')]
        out_dic[unicode('token', 'utf-8')] = dic[unicode('token', 'utf-8')]
        out_dic[unicode('tokenid', 'utf-8')] = dic[unicode('tokenid', 'utf-8')]


        new_dic[id] = {unicode('input', 'utf-8'):[in_dic], unicode('output', 'utf-8'):[out_dic],
            unicode('utt2spk', 'utf-8'):dic[unicode('utt2spk', 'utf-8')]}
        
    jsonstring = json.dumps({'utts': new_dic}, indent=4, ensure_ascii=False).encode('utf_8')
    print(jsonstring)

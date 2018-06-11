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
    parser.add_argument('json', type=str,
                        help='json file')
    parser.add_argument('--multi', '-m', type=int,
                        help='Test the json file for multiple input/output', default=0)
    args = parser.parse_args()
    
    # logging info
    logging.basicConfig(level=logging.INFO, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")

    j = json.load(open(args.json))
    old_dic = j['utts']

    new_dic = {}
    for item in old_dic.items():
        id, dic = item

        in_dic = {}
        if dic.has_key(unicode('idim', 'utf-8')):
            in_dic[unicode('shape', 'utf-8')] = (int(dic[unicode('ilen', 'utf-8')]), int(dic[unicode('idim', 'utf-8')]))
        in_dic[unicode('name', 'utf-8')] = unicode('input1', 'utf-8')
        in_dic[unicode('feat', 'utf-8')] = dic[unicode('feat', 'utf-8')]

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

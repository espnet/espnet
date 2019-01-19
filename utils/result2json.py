#!/usr/bin/env python
# encoding: utf-8

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#           2018 Xuankai Chang (Shanghai Jiao Tong University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import json
import re
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--key', '-k', type=str,
                        help='key')
    args = parser.parse_args()

    key = re.findall(r"r\d+h\d+", args.key)[0]

    re_id = r'^id: '
    re_strings = {'Speaker': r'^Speaker sentences',
                  'Scores': r'^Scores: ',
                  'REF': r'^REF: ',
                  'HYP': r'^HYP: '}
    re_id = re.compile(re_id)
    re_patterns = {}
    for p in re_strings.keys():
        re_patterns[p] = re.compile(re_strings[p])

    ret = {}
    tmp_id = None
    tmp_ret = {}

    line = sys.stdin.readline()
    while line:
        x = unicode(line, 'utf_8').rstrip()
        x_split = x.split()

        if re_id.match(x):
            if tmp_id:
                ret[tmp_id.encode('utf_8')] = {key:tmp_ret}
                tmp_ret = {}
            tmp_id = x_split[1]
        for p in re_patterns.keys():
            if re_patterns[p].match(x):
                tmp_ret[p] = ' '.join(x_split[1:]).encode('utf_8')

        line = sys.stdin.readline()

    if tmp_ret != {}:
        ret[tmp_id.encode('utf_8')] = {key:tmp_ret}

    all_l = {'utts': ret}
    # ensure "ensure_ascii=False", which is a bug
    jsonstring = json.dumps(all_l, indent=4, ensure_ascii=False)
    print(jsonstring)

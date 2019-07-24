#!/usr/bin/env python
# encoding: utf-8

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import print_function
from __future__ import unicode_literals

import argparse
from io import open
import json
import logging
import re


PY2 = sys.version_info[0] == 2
sys.stdin = codecs.getreader('utf-8')(
    sys.stdin if PY2 else sys.stdin.buffer)
sys.stdout = codecs.getwriter('utf-8')(
    sys.stdout if PY2 else sys.stdout.buffer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('jsons', type=str, nargs='+',
                        help='json files')
    parser.add_argument('--filt', '-f', type=str,
                        help='utterance ID filter')
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

    # make union set for utterance keys
    new_dic = dict()
    for x in args.jsons:        #
        with open(x, 'r', encoding='utf_8') as f:
            j = json.load(f)
        ks = j['utts'].keys()
        logging.info(x + ': has ' + str(len(ks)) + ' utterances')
        for k in ks:
            if re.search(args.filt, k):
                new_dic[k] = j['utts'][k]
    logging.info('new json has ' + str(len(new_dic.keys())) + ' utterances')

    # ensure "ensure_ascii=False", which is a bug
    jsonstring = json.dumps({'utts': new_dic}, indent=4, ensure_ascii=False, sort_keys=True)
    print(jsonstring)

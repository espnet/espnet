#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
from __future__ import unicode_literals

import argparse
import codecs
import json
import logging

from espnet.utils.cli_utils import get_commandline_args


def get_parser():
    parser = argparse.ArgumentParser(
        description='convert ASR recognized json to text',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('json', type=str, help='json files')
    parser.add_argument('dict', type=str, help='dict')
    parser.add_argument('ref', type=str, help='ref')
    parser.add_argument('hyp', type=str, help='hyp')
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()

    # logging info
    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=logfmt)
    logging.info(get_commandline_args())

    logging.info("reading %s", args.json)
    with codecs.open(args.json, 'r', encoding="utf-8") as f:
        j = json.load(f)

    logging.info("reading %s", args.dict)
    with codecs.open(args.dict, 'r', encoding="utf-8") as f:
        dictionary = f.readlines()
    char_list = [entry.split(' ')[0] for entry in dictionary]
    char_list.insert(0, '<blank>')
    char_list.append('<eos>')
    # print([x.encode('utf-8') for x in char_list])

    logging.info("writing hyp trn to %s", args.hyp)
    logging.info("writing ref trn to %s", args.ref)
    h = codecs.open(args.hyp, 'w', encoding="utf-8")
    r = codecs.open(args.ref, 'w', encoding="utf-8")

    for x in j['utts']:
        seq = [char_list[int(i)] for i in j['utts'][x]['output'][0]['rec_tokenid'].split()]
        h.write(x + " " + " ".join(seq).replace('<eos>', '') + '\n')

        if 'tokenid' in j['utts'][x]['output'][0].keys():
            seq = [char_list[int(i)] for i in j['utts'][x]['output'][0]['tokenid'].split()]
            r.write(x + " " + " ".join(seq).replace('<eos>', '') + '\n')

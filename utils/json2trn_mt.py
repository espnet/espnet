#!/usr/bin/env python
# encoding: utf-8

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# NOTE: this is made for machine translation

from __future__ import unicode_literals

import argparse
import codecs
import json
import logging

from espnet.utils.cli_utils import get_commandline_args


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('json', type=str, help='json files')
    parser.add_argument('dict', type=str, help='dict for target language')
    parser.add_argument('ref', type=str, help='ref')
    parser.add_argument('--hyp', type=str, help='hyp', default=False, nargs='?')
    parser.add_argument('--src', type=str, help='src', default=False, nargs='?')
    parser.add_argument('--dict-src', type=str, help='dict for source language',
                        default=False, nargs='?')
    args = parser.parse_args()

    # logging info
    logfmt = '%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s'
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

    if args.dict_src:
        logging.info("reading %s", args.dict_src)
        with codecs.open(args.dict_src, 'r') as f:
            dictionary_src = f.readlines()
        char_list_src = [unicode(entry.split(' ')[0], 'utf_8') for entry in dictionary_src]
        char_list_src.insert(0, '<blank>')
        char_list_src.append('<eos>')

    if args.hyp:
        logging.info("writing hyp trn to %s", args.hyp)
        h = codecs.open(args.hyp, 'w', encoding="utf-8")

    logging.info("writing ref trn to %s", args.ref)
    r = codecs.open(args.ref, 'w', encoding="utf-8")

    if args.src:
        logging.info("writing src trn to %s", args.src)
        s = codecs.open(args.src, 'w', encoding="utf-8")

    for x in j['utts']:
        if args.hyp:
            seq = [char_list[int(i)] for i in j['utts'][x]['output'][0]['rec_tokenid'].split()]
            h.write(" ".join(seq).replace('<eos>', '')),
            h.write(" (" + j['utts'][x]['utt2spk'].replace('-', '_') + "-" + x + ")\n")

        seq = [char_list[int(i)] for i in j['utts'][x]['output'][0]['tokenid'].split()]
        r.write(" ".join(seq).replace('<eos>', '')),
        r.write(" (" + j['utts'][x]['utt2spk'].replace('-', '_') + "-" + x + ")\n")

        if args.src and 'tokenid_src' in j['utts'][x]['output'][0].keys():
            if args.dict_src:
                seq = [char_list_src[int(i)] for i in j['utts'][x]['output'][0]['tokenid_src'].split()]
            else:
                seq = [char_list[int(i)] for i in j['utts'][x]['output'][0]['tokenid_src'].split()]
            s.write(" ".join(seq).replace('<eos>', '')),
            s.write(" (" + j['utts'][x]['utt2spk'].replace('-', '_') + "-" + x + ")\n")

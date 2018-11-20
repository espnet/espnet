#!/usr/bin/env python2
# encoding: utf-8

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import json
import argparse
import logging

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('json', type=str, help='json files')
    parser.add_argument('dict', type=str, help='dict')
    parser.add_argument('ref', type=str, help='ref')
    parser.add_argument('--hyp', type=str, help='hyp', default=False)
    parser.add_argument('--src', type=str, help='src', default=False)
    args = parser.parse_args()

    # logging info
    logging.basicConfig(level=logging.INFO, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")

    logging.info("reading %s", args.json)
    with open(args.json, 'r') as f:
        j = json.load(f)

    logging.info("reading %s", args.dict)
    with open(args.dict, 'r') as f:
        dictionary = f.readlines()
    char_list = [unicode(entry.split(' ')[0], 'utf_8') for entry in dictionary]
    char_list.insert(0, '<blank>')
    char_list.append('<eos>')

    if args.hyp:
        logging.info("writing hyp trn to %s", args.hyp)
        h = open(args.hyp, 'w')

    logging.info("writing ref trn to %s", args.ref)
    r = open(args.ref, 'w')

    if args.src:
        logging.info("writing src trn to %s", args.src)
        s = open(args.src, 'w')

    for x in j['utts']:
        if args.hyp:
            seq = [char_list[int(i)] for i in j['utts'][x]['output'][0]['rec_tokenid'].split()]
            h.write(" ".join(seq).encode('utf-8').replace('<eos>', '')),
            h.write(" (" + j['utts'][x]['utt2spk'].replace('-', '_') + "-" + x + ")\n")

        seq = [char_list[int(i)] for i in j['utts'][x]['output'][0]['tokenid'].split()]
        r.write(" ".join(seq).encode('utf-8').replace('<eos>', '')),
        r.write(" (" + j['utts'][x]['utt2spk'].replace('-', '_') + "-" + x + ")\n")

        if args.src and 'tokenid_src' in j['utts'][x]['output'][0].keys():
            seq = [char_list[int(i)] for i in j['utts'][x]['output'][0]['tokenid_src'].split()]
            s.write(" ".join(seq).encode('utf-8').replace('<eos>', '')),
            s.write(" (" + j['utts'][x]['utt2spk'].replace('-', '_') + "-" + x + ")\n")

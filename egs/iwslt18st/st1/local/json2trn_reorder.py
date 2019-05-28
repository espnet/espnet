#!/usr/bin/env python
# encoding: utf-8

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import json
import argparse
import codecs
import logging

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('json', type=str, help='json files')
    parser.add_argument('dict', type=str, help='dict')
    parser.add_argument('hyp', type=str, help='hyp')
    parser.add_argument('file_order', type=str,
                        help='text file which describes the order of audio files')
    args = parser.parse_args()

    # logging info
    logging.basicConfig(level=logging.INFO, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")

    file_order = []
    with codecs.open(args.file_order, 'r', encoding="utf-8") as f:
        for line in f:
            file_order.append(line.strip())

    logging.info("reading %s", args.json)
    with codecs.open(args.json, 'r', encoding="utf-8") as f:
        j = json.load(f)

    logging.info("reading %s", args.dict)
    with codecs.open(args.dict, 'r', encoding="utf-8") as f:
        dictionary = f.readlines()
    char_list = [entry.split(' ')[0] for entry in dictionary]
    char_list.insert(0, '<blank>')
    char_list.append('<eos>')

    logging.info("writing hyp trn to %s", args.hyp)
    h = codecs.open(args.hyp, 'w', encoding="utf-8")

    hyps = {}
    for x in j['utts']:
        talkid = x.split('_')[0]
        start_time = int(x.split('_')[1])
        if talkid not in hyps.keys():
            hyps[talkid] = {}

        seq = [char_list[int(i)] for i in j['utts'][x]['output'][0]['rec_tokenid'].split()]
        hyps[talkid][start_time] = [x, seq]

    for talkid in file_order:
        for start_time, (x, seq) in sorted(hyps[talkid].items(), key=lambda x: x[0]):
            h.write(" ".join(seq).replace('<eos>', '')),
            h.write(" (" + j['utts'][x]['utt2spk'].replace('-', '_') + "-" + x + ")\n")

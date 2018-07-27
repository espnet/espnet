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
    parser.add_argument('hyp', type=str, help='hyp')
    parser.add_argument('output_type', type=str, default="grapheme", help='output type ("grapheme" or "phn")')
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
    # print([x.encode('utf-8') for x in char_list])

    logging.info("writing hyp trn to %s", args.hyp)
    logging.info("writing ref trn to %s", args.ref)
    h = open(args.hyp, 'w')
    r = open(args.ref, 'w')

    for x in j['utts']:
        found_output = False
        for output in j['utts'][x]['output']:
            if output['name'] == args.output_type:
                found_output = True
                seq = [char_list[int(i)] for i in output['rec_tokenid'].split()]
                h.write(" ".join(seq).encode('utf-8').replace('<eos>', '')),
                h.write(" (" + j['utts'][x]['utt2spk'].replace('-', '_') + "-" + x +")\n")

                seq = [char_list[int(i)] for i in output['tokenid'].split()]
                r.write(" ".join(seq).encode('utf-8').replace('<eos>', '')),
                r.write(" (" + j['utts'][x]['utt2spk'].replace('-', '_') + "-" + x +")\n")
        if not found_output:
            logging.warning("Didn't find output type: {}".format(args.output_type))

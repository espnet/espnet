#!/usr/bin/env python2
# encoding: utf-8

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#           2018 Xuankai Chang (Shanghai Jiao Tong University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import json
import argparse
import logging

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('json', type=str, help='json files')
    parser.add_argument('dict', type=str, help='dict')
    parser.add_argument('--num_spkrs', type=int, help='number of speakers')
    parser.add_argument('--refs', type=str, nargs='+', help='ref for all speakers')
    parser.add_argument('--hyps', type=str, nargs='+', help='hyp for all outputs')
    args = parser.parse_args()

    n_ref = len(args.refs)
    n_hyp = len(args.hyps)
    assert n_ref == n_hyp
    assert n_ref == args.num_spkrs

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

    hyps = []
    refs = []
    for ns in range(args.num_spkrs):
        hyp_file = open(args.hyps[ns], 'w')
        ref_file = open(args.refs[ns], 'w')

        for x in j['utts']:
            # hyps
            if args.num_spkrs == 1:
                seq = [char_list[int(i)] for i in j['utts'][x]['output'][0]['rec_tokenid'].split()]
            else:
                seq = [char_list[int(i)] for i in j['utts'][x]['output'][ns][0]['rec_tokenid'].split()]
            hyp_file.write(" ".join(seq).encode('utf-8').replace('<eos>', '')),
            hyp_file.write(" (" + j['utts'][x]['utt2spk'].replace('-', '_') + "-" + x +")\n")

            # ref
            if args.num_spkrs == 1:
                seq = [char_list[int(i)] for i in j['utts'][x]['output'][0]['tokenid'].split()]
            else:
                seq = [char_list[int(i)] for i in j['utts'][x]['output'][ns][0]['tokenid'].split()]
            ref_file.write(" ".join(seq).encode('utf-8').replace('<eos>', '')),
            ref_file.write(" (" + j['utts'][x]['utt2spk'].replace('-', '_') + "-" + x +")\n")

        hyp_file.close()
        ref_file.close()

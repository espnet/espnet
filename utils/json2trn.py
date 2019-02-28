#!/usr/bin/env python
# encoding: utf-8

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#           2018 Xuankai Chang (Shanghai Jiao Tong University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import codecs
import json
import logging
import sys

from espnet.utils.cli_utils import get_commandline_args


def main(args):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('json', type=str, help='json files')
    parser.add_argument('dict', type=str, help='dict')
    parser.add_argument('--num-spkrs', type=int, default=1, help='number of speakers')
    parser.add_argument('--refs', type=str, nargs='+', help='ref for all speakers')
    parser.add_argument('--hyps', type=str, nargs='+', help='hyp for all outputs')
    args = parser.parse_args(args)

    n_ref = len(args.refs)
    n_hyp = len(args.hyps)
    assert n_ref == n_hyp
    assert n_ref == args.num_spkrs

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

    hyps = []
    refs = []
    for ns in range(args.num_spkrs):
        hyp_file = codecs.open(args.hyps[ns], 'w', encoding="utf-8")
        ref_file = codecs.open(args.refs[ns], 'w', encoding="utf-8")

        for x in j['utts']:
            # hyps
            if args.num_spkrs == 1:
                seq = [char_list[int(i)] for i in j['utts'][x]['output'][0]['rec_tokenid'].split()]
            else:
                seq = [char_list[int(i)] for i in j['utts'][x]['output'][ns][0]['rec_tokenid'].split()]
            hyp_file.write(" ".join(seq).replace('<eos>', '')),
            hyp_file.write(" (" + j['utts'][x]['utt2spk'].replace('-', '_') + "-" + x + ")\n")

            # ref
            if args.num_spkrs == 1:
                seq = [char_list[int(i)] for i in j['utts'][x]['output'][0]['tokenid'].split()]
            else:
                seq = [char_list[int(i)] for i in j['utts'][x]['output'][ns][0]['tokenid'].split()]
            ref_file.write(" ".join(seq).replace('<eos>', '')),
            ref_file.write(" (" + j['utts'][x]['utt2spk'].replace('-', '_') + "-" + x + ")\n")

        hyp_file.close()
        ref_file.close()


if __name__ == '__main__':
    main(sys.argv[1:])

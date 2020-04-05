#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# NOTE: this is made for machine translation

import argparse
import codecs
import json
import logging
import sys

from espnet.utils.cli_utils import get_commandline_args


def get_parser():
    parser = argparse.ArgumentParser(
        description='convert json to machine translation transcription',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('json', type=str, help='json files')
    parser.add_argument('dict', type=str, help='dict for target language')
    parser.add_argument('--refs', type=str, nargs='+', help='ref for all speakers')
    parser.add_argument('--hyps', type=str, nargs='+', help='hyp for all outputs')
    parser.add_argument('--srcs', type=str, nargs='+', help='src for all outputs')
    parser.add_argument('--dict-src', type=str, help='dict for source language',
                        default=False, nargs='?')
    return parser


def main(args):
    parser = get_parser()
    args = parser.parse_args(args)
    convert(args.json, args.dict, args.refs, args.hyps, args.srcs, args.dict_src)


def convert(jsonf, dic, refs, hyps, srcs, dic_src):

    # logging info
    logfmt = '%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s'
    logging.basicConfig(level=logging.INFO, format=logfmt)
    logging.info(get_commandline_args())

    logging.info("reading %s", jsonf)
    with codecs.open(jsonf, 'r', encoding="utf-8") as f:
        j = json.load(f)

    # target dictionary
    logging.info("reading %s", dic)
    with codecs.open(dic, 'r', encoding="utf-8") as f:
        dictionary = f.readlines()
    char_list_tgt = [entry.split(' ')[0] for entry in dictionary]
    char_list_tgt.insert(0, '<blank>')
    char_list_tgt.append('<eos>')

    # source dictionary
    logging.info("reading %s", dic_src)
    if dic_src:
        with codecs.open(dic_src, 'r', encoding="utf-8") as f:
            dictionary = f.readlines()
        char_list_src = [entry.split(' ')[0] for entry in dictionary]
        char_list_src.insert(0, '<blank>')
        char_list_src.append('<eos>')

    if hyps:
        hyp_file = codecs.open(hyps[0], 'w', encoding="utf-8")
    ref_file = codecs.open(refs[0], 'w', encoding="utf-8")
    if srcs:
        src_file = codecs.open(srcs[0], 'w', encoding="utf-8")

    for x in j['utts']:
        # hyps
        if hyps:
            seq = [char_list_tgt[int(i)] for i in j['utts'][x]['output'][0]['rec_tokenid'].split()]
            hyp_file.write(" ".join(seq).replace('<eos>', '')),
            hyp_file.write(" (" + j['utts'][x]['utt2spk'].replace('-', '_') + "-" + x + ")\n")

        # ref
        seq = [char_list_tgt[int(i)] for i in j['utts'][x]['output'][0]['tokenid'].split()]
        ref_file.write(" ".join(seq).replace('<eos>', '')),
        ref_file.write(" (" + j['utts'][x]['utt2spk'].replace('-', '_') + "-" + x + ")\n")

        # src
        if 'tokenid_src' in j['utts'][x]['output'][0].keys():
            if dic_src:
                seq = [char_list_src[int(i)] for i in j['utts'][x]['output'][0]['tokenid_src'].split()]
            else:
                seq = [char_list_tgt[int(i)] for i in j['utts'][x]['output'][0]['tokenid_src'].split()]
            src_file.write(" ".join(seq).replace('<eos>', '')),
            src_file.write(" (" + j['utts'][x]['utt2spk'].replace('-', '_') + "-" + x + ")\n")

    if hyps:
        hyp_file.close()
    ref_file.close()
    if srcs:
        src_file.close()


if __name__ == '__main__':
    main(sys.argv[1:])

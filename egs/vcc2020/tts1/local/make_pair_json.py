#!/usr/bin/env python3
# encoding: utf-8

import argparse
import codecs
from distutils.util import strtobool
from io import open
import json
import logging
import sys
import os

from espnet.utils.cli_utils import get_commandline_args
from espnet.utils.cli_readers import file_reader_helper

def read_shape(scp):
    return {utt_id:shape for idx, (utt_id, shape) in enumerate(file_reader_helper("scp:" + scp, return_shape=True), 1)}
    

def get_parser():
    parser = argparse.ArgumentParser(
        description='Merge source and target data.json files into one json file.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--src_json', type=str,
                        help='Json file for the source speaker')
    parser.add_argument('--trg_json', type=str,
                        help='Json file for the target speaker')
    parser.add_argument('--src_decoded_feat', type=str,
                        help='scp file of the TTS generated features using the source TTS')
    parser.add_argument('--trg_decoded_feat', type=str,
                        help='scp file of the TTS generated features using the target TTS')
    parser.add_argument('--pwd', type=str,
                        help='current working directory')
    parser.add_argument('--verbose', '-V', default=1, type=int,
                        help='Verbose option')
    parser.add_argument('--out', '-O', type=str,
                        help='The output filename. '
                             'If omitted, then output to sys.stdout')
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    # logging info
    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    if args.verbose > 0:
        logging.basicConfig(level=logging.INFO, format=logfmt)
    else:
        logging.basicConfig(level=logging.WARN, format=logfmt)
    logging.info(get_commandline_args())

    with open(args.src_json, 'rb') as f:
        src_json = json.load(f)['utts']
    with open(args.trg_json, 'rb') as f:
        trg_json = json.load(f)['utts']

    
    src_decoded_feat = {line.rstrip().split(' ')[0]:line.rstrip().split(' ')[1] for line in open(args.src_decoded_feat, 'r')}
    src_decoded_feat_shape = read_shape(args.src_decoded_feat)
    trg_decoded_feat = {line.rstrip().split(' ')[0]:line.rstrip().split(' ')[1] for line in open(args.trg_decoded_feat, 'r')}
    trg_decoded_feat_shape = read_shape(args.trg_decoded_feat)

    # get source and target speaker
    _ = list(src_json.keys())[0].split('_')
    srcspk = _[0]
    _ = list(trg_json.keys())[0].split('_')
    trgspk = _[0]

    data = {"utts" : {} }
    
    for k, v in src_json.items():

        inp = v['input']
        out = [{
            "feat": os.path.join(args.pwd, trg_decoded_feat[k]),
            "name": "target1",
            "shape": trg_decoded_feat_shape[k],
        }]

        entry = {"input" : inp,
                 "output" : out,
                 }
        data["utts"][k] = entry
    
    for k, v in trg_json.items():

        out = v['input']
        inp = [{
            "feat": os.path.join(args.pwd, src_decoded_feat[k]),
            "name": "input1",
            "shape": src_decoded_feat_shape[k],
        }]

        entry = {"input" : inp,
                 "output" : out,
                 }
        data["utts"][k] = entry
    
    if args.out is None:
        out = sys.stdout
    else:
        out = open(args.out, 'w', encoding='utf-8')
        
    json.dump(data, out,
               indent=4, ensure_ascii=False,
               separators=(',', ': '),
               )

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert given text data to json format required by ESPNet
"""
import argparse
import json
import os
from typing import Dict, List

from logging import getLogger

logger = getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser(description='generate json file')
    parser.add_argument('--src', '-s', type=os.path.abspath, required=True, help='path to source language data')
    parser.add_argument('--trg', '-t', type=os.path.abspath, required=True, help='path to target language data')
    parser.add_argument('--src-vocab', '-sv', type=os.path.abspath, required=True, help='path to source vocabulary')
    parser.add_argument('--trg-vocab', '-tv', type=os.path.abspath, required=True, help='path to target vocabulary')
    parser.add_argument('--dest', '-d', type=os.path.abspath, default='data.json', help='path to output json file')
    args = parser.parse_args()
    return args


def load_vocab_file(path: str) -> Dict[str, int]:
    vocab: Dict[str, int] = {}
    with open(path, 'r') as fi:
        for line in fi:
            token, index = line.strip().split(' ')
            vocab[token] = int(index)
    return vocab


def convert_line_to_dict(line: str, vocab: dict, name: str) -> Dict:
    tokens: List = line.strip().split()
    out: Dict = {
        'name': name,
        'shape': [len(tokens), len(vocab) + 2],
        'token': ' '.join(tokens),
        'tokenid': ' '.join([str(vocab[t]) if t in vocab else str(vocab['<unk>']) for t in tokens])
    }
    return out


def merge_src_and_trg_to_utts(src_dicts: List, trg_dicts: List, name: str = 'iwslt') -> List:
    output_list: List = []
    for n, (src_dict, trg_dict) in enumerate(zip(src_dicts, trg_dicts)):
        out: Dict = {
            'input': [],
            'output': [
                trg_dict,
                src_dict
            ],
            'utt2spk': "{}_{}".format(name, n)
        }
        output_list.append(('{}_{}'.format(name, n), out))
    return output_list


def main(args):
    logger.info(args)
    src_vocab: Dict = load_vocab_file(args.src_vocab)
    trg_vocab: Dict = load_vocab_file(args.trg_vocab)

    source_dicts: List = [convert_line_to_dict(l, vocab=src_vocab, name='target2') for l in open(args.src)]
    target_dicts: List = [convert_line_to_dict(l, vocab=trg_vocab, name='target1') for l in open(args.trg)]

    utt_list: List = merge_src_and_trg_to_utts(source_dicts, target_dicts, name='iwslt')
    output: dict = {
        'utts': {}
    }
    for name, utt in utt_list:
        output['utts'][name] = utt

    logger.info('Saved to {}'.format(args.dest))
    with open(args.dest, 'w') as fo:
        json.dump(output, fo, indent=4)


if __name__ == "__main__":
    args = get_args()
    main(args)

#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
from __future__ import unicode_literals

import argparse
from os.path import join
import os
import sys


def get_parser():
    parser = argparse.ArgumentParser(
        description='Convert text script file to its shape using TTS frontend',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('scp', type=str, help='Input script file')
    parser.add_argument('--num-vocab', default=-1, type=int,
                        help='Number of vocabulary')
    parser.add_argument('--frontend', type=str, choices=["en", "text"],
                        help="Text tokenization frontend.")
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    scp_file = args.scp
    frontend = args.frontend
    num_vocab = args.num_vocab
    if frontend is not None and frontend != "":
        from espnet.tts import frontend as fe
        # TODO: pass frontend_conf or similar
        frontend = fe.get_frontend(frontend)
        num_vocab = frontend.num_vocab()
    else:
        frontend = None

    with open(scp_file) as f:
        for l in f:
            utt_id, text = l.strip().split(" ", 1)
            if frontend is not None:
                text = frontend.text_to_sequence(text)
                if isinstance(text, tuple):
                    text = text[0]
            sys.stdout.write(f"{utt_id} {len(text)},{num_vocab}\n")

    sys.exit(0)

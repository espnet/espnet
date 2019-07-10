#!/usr/bin/env python
# coding: utf-8
"""Text script file to its shape using TTS frontend.

usage: text2shape.py [options] <scp>

options:
    --frontend=<f>           Text tokenization frontend.
    --num-vocab=<N>  　　　　　Num vocab [default: -1].
    -h, --help               Show help message.
"""
from docopt import docopt
import sys
import numpy as np
from os.path import join
import os
import sys

if __name__ == "__main__":
    args = docopt(__doc__)
    scp_file = args["<scp>"]
    frontend = args["--frontend"]
    num_vocab = args["--num-vocab"]
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

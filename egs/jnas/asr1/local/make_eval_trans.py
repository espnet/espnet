#!/usr/bin/env python3
# Copyright 2012 Vassil Panayotov
# Apache 2.0

"""
Takes a "PROMPTS" file with lines like:
1snoke-20120412-hge/mfc/a0405 IT SEEMED THE ORDAINED ORDER OF THINGS THAT DOGS SHOULD WORK

, an ID prefix and a list of audio file names (e.g. for above example the list will contain "a0405").
It checks if the prompts file have transcription for all audio files in the list and
if this is the case produces a transcript line for each file in the format:
prefix_a0405 IT SEEMED THE ORDAINED ORDER OF THINGS THAT DOGS SHOULD WORK
"""
from __future__ import print_function

import sys

def err(msg):
    print(msg, file=sys.stderr)

if len(sys.argv) < 3:
    err("Usage: %s <prompts-file> <id-prefix> <utt-id1> <utt-id2> ... " % sys.argv[0])
    sys.exit(1)

#err(str(sys.argv))
id_prefix = sys.argv[2:]
utt2trans = dict()

for l in open(sys.argv[1], 'r', encoding='utf-8'):
    u, trans = l.split(None, 1)
    utt2trans[u] = trans.strip('\n')

for uid in id_prefix:
    if not uid.split('_')[-1] in utt2trans:
        err("No transcript found for %s" %(uid))
        continue
    print("%s %s" % (uid, utt2trans[uid.split('_')[-1]]))

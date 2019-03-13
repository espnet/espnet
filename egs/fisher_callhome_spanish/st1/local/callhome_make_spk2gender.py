#!/usr/bin/env python
# encoding: utf-8

# Note(kamo-naoyuki) 31,Jan,2019:
# This file is copied from kaldi/egs/fisher_callhome_spanish/s5/local/callhole_make_spk2gender.sh
# and modified for py2/3 compatibility.

# Copyright 2014  Gaurav Kumar.   Apache 2.0
# Gets the unique speakers from the file created by fsp_make_trans.pl
# Note that if a speaker appears multiple times, it is categorized as female

from __future__ import print_function
from __future__ import unicode_literals

import codecs
from io import open
import sys

PY2 = sys.version_info[0] == 2
sys.stdin = codecs.getreader('utf-8')(
    sys.stdin if PY2 else sys.stdin.buffer)
sys.stdout = codecs.getwriter('utf-8')(
    sys.stdout if PY2 else sys.stdout.buffer)


if __name__ == '__main__':
    tmpFileLocation = 'data/local/tmp/callhome_spk2gendertmp'

    tmpFile = None

    try:
        tmpFile = open(tmpFileLocation, encoding='utf-8')
    except IOError:
        print('The file spk2gendertmp does not exist. Run fsp_make_trans.pl first?',
              file=sys.stderr)
        raise

    speakers = {}

    for line in tmpFile:
        comp = line.split(' ')
        if comp[0] in speakers:
            speakers[comp[0]] = "f"
        else:
            speakers[comp[0]] = comp[1]

    for speaker, gender in speakers.items():
        print(speaker + " " + gender)

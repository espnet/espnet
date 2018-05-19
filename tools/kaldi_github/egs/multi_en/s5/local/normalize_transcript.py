#!/usr/bin/env python

# Copyright 2016  Allen Guo
#           2017  Xiaohui Zhang
# Apache License 2.0

# This script normalizes the given "text" (transcript) file. The normalized result
# is printed to STDOUT. This normalization should be applied to all corpora.

import re
import sys

def normalize(utt):
    utt = utt.lower() \
             .replace('[uh]', 'uh') \
             .replace('[um]', 'um') \
             .replace('<noise>', '[noise]') \
             .replace('<spoken_noise>', '[vocalized-noise]') \
             .replace('.period', 'period') \
             .replace('.dot', 'dot') \
             .replace('-hyphen', 'hyphen') \
             .replace('._', '. ') \
             .translate(None, '()*;:"!&{},')
    utt = re.sub(r"'([a-z]+)'", r'\1', utt)  # Unquote quoted words
    return utt

def main():
    if len(sys.argv) != 2:
        print 'Usage: local/normalize_transcript.py [text_file]'
        sys.exit(1)
    with open(sys.argv[1], 'r') as f:
        for line in f.readlines():
            chunks = line.split(' ')
            if len(chunks) > 1:
                sys.stdout.write(chunks[0].strip() + ' ' + normalize(' '.join(chunks[1:])))
            else:
                sys.stdout.write(chunks[0].strip() + '\n')

if __name__ == '__main__':
    main()

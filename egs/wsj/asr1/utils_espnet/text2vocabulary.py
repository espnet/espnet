#!/usr/bin/env python

# Copyright 2018 Mitsubishi Electric Research Laboratories (Takaaki Hori)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import sys
import six
import logging

##################################
# main
if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', '-o', default='', type=str,
                        help='output a vocabulary file')
    parser.add_argument('--cutoff', '-c', default=0, type=int,
                        help='cut-off frequency')
    parser.add_argument('--vocabsize', '-s', default=20000, type=int,
                        help='vocabulary size')
    parser.add_argument('text_files', nargs='*',
                        help='input text files')
    args = parser.parse_args()

    # count the word occurrences
    counts = {}
    exclude = ['<sos>', '<eos>', '<unk>']
    if len(args.text_files) == 0:
        args.text_files.append('-')
    for fn in args.text_files:
        fd = open(fn, 'r') if fn != '-' else sys.stdin
        for ln in fd.readlines():
            for tok in ln.split():
                if tok not in exclude:
                    if tok not in counts:
                        counts[tok] = 1
                    else:
                        counts[tok] += 1
        if fn != '-':
            fd.close()

    # limit the vocabulary size
    total_count = sum(counts.values())
    invocab_count = 0
    vocabulary = []
    for w,c in sorted(counts.items(), key=lambda x:-x[1]):
        if c <= args.cutoff:
            break
        if len(vocabulary) >= args.vocabsize:
            break
        vocabulary.append(w)
        invocab_count += c

    logging.warn('OOV rate = %.2f %%' % (float(total_count - invocab_count)/total_count*100))
    # write the vocabulary
    fd = open(args.output, 'w') if args.output else sys.stdout
    six.print_('<unk> 1', file=fd)
    for n,w in enumerate(sorted(vocabulary)):
        six.print_('%s %d' % (w, n+2), file=fd)
    if args.output:
        fd.close()

